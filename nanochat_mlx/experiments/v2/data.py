"""Document-disjoint local corpus, BOS best-fit replay with provenance.

The same document hash always maps to one split. Duplicate occurrences remain
in that split, preserving frequency. Reserved token arrays are never opened by
the search worker. Source shards and tokenizer are recorded by content hash.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from .records import digest,write_json


def split_for(document_id,salt):
    bucket=int(hashlib.sha256((salt+document_id).encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<90 else 'calibration' if bucket<95 else 'reserved'


def packed_rows(documents,tokenizer,length,rows,buffer_size=1000):
    """The original largest-fit/shortest-crop policy, retaining segment IDs."""
    buffer=[];iterator=iter(documents)
    for row_number in range(rows):
        row=[];segments=[]
        while len(row)<length+1:
            while len(buffer)<buffer_size:
                group=[]
                for _ in range(128):
                    try:group.append(next(iterator))
                    except StopIteration:break
                if not group:raise ValueError('Insufficient training documents for replay; no silent cycling')
                encoded=tokenizer.encode([d[2] for d in group],prepend=tokenizer.get_bos_token_id())
                buffer.extend((d[0],d[1],t) for d,t in zip(group,encoded))
            remaining=length+1-len(row)
            fitting=[i for i,d in enumerate(buffer) if len(d[2])<=remaining]
            index=max(fitting,key=lambda i:len(buffer[i][2])) if fitting else min(range(len(buffer)),key=lambda i:len(buffer[i][2]))
            doc_id,occurrence,tokens=buffer.pop(index)
            used=tokens[:remaining];start=len(row);row.extend(used)
            segments.append({'document_id':doc_id,'occurrence':occurrence,'row_start':start,
                'row_end':len(row),'document_token_start':0,'document_token_end':len(used),
                'cropped_tokens':len(tokens)-len(used)})
        yield np.array(row,np.int32),{'row':row_number,'segments':segments}


def prepare(config,root):
    import pyarrow.parquet as pq
    os.environ['NANOCHAT_BASE_DIR']=str(Path(config['data_dir']).expanduser())
    from nanochat_mlx.tokenizer import get_tokenizer
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    sources=sorted((Path(config['data_dir']).expanduser()/'base_data').glob('*.parquet'))
    if not sources:raise FileNotFoundError('No local parquet corpus')
    con=sqlite3.connect(root/'documents.sqlite')
    con.execute('create table documents (occurrence integer primary key, id text, split text, text text, source text, source_row integer)')
    source_hashes={};counts={'train':0,'calibration':0,'reserved':0}
    for source in sources:
        source_hashes[str(source.resolve())]=digest(source);source_row=0
        for batch in pq.ParquetFile(source).iter_batches(columns=['text'],batch_size=1024):
            rows=[]
            for text in batch.column(0).to_pylist():
                if not isinstance(text,str):raise ValueError('Non-text corpus row')
                doc_id=hashlib.sha256(text.encode()).hexdigest();split=split_for(doc_id,config['split_salt'])
                rows.append((doc_id,split,text,source.name,source_row));source_row+=1;counts[split]+=1
            con.executemany('insert into documents(id,split,text,source,source_row) values(?,?,?,?,?)',rows)
        con.commit();print('indexed',source.name,source_row,flush=True)
    con.execute('create index document_split on documents(split,occurrence)')
    con.execute('create index document_hash on documents(id)');con.commit()
    unique=con.execute('select count(distinct id) from documents').fetchone()[0]
    overlap=con.execute('select count(*) from (select id from documents group by id having count(distinct split)>1)').fetchone()[0]
    if overlap:raise ValueError('Document split overlap')
    tok=get_tokenizer();length=config['sequence_len'];batch=config['device_batch_size']
    n_batches=(config['reference_steps']+100)*config['accumulation']
    x=np.lib.format.open_memmap(root/'train_x.npy',mode='w+',dtype=np.int32,shape=(n_batches,batch,length))
    y=np.lib.format.open_memmap(root/'train_y.npy',mode='w+',dtype=np.int32,shape=x.shape)
    docs=con.execute("select id,occurrence,text from documents where split='train' order by occurrence")
    with (root/'train_provenance.jsonl').open('x') as f:
        for i,(row,provenance) in enumerate(packed_rows(docs,tok,length,n_batches*batch)):
            x[i//batch,i%batch]=row[:-1];y[i//batch,i%batch]=row[1:]
            f.write(json.dumps(provenance,separators=(',',':'))+'\n')
            if i%1000==0:print('replay rows',i,'/',n_batches*batch,flush=True)
    x.flush();y.flush();del x,y
    quality_meta={}
    for split,key in [('calibration','calibration_sequences'),('reserved','reserved_sequences')]:
        # One sequence per distinct document. Padding targets are ignored.
        docs=con.execute('select id,min(occurrence),text from documents where split=? group by id order by min(occurrence) limit ?', (split,config[key]))
        qx=[];qy=[];meta=[]
        for doc_id,occurrence,text in docs:
            tokens=tok.encode(text,prepend=tok.get_bos_token_id())[:length+1]
            if len(tokens)<2:continue
            a=np.full(length,tok.get_bos_token_id(),np.int32);b=np.full(length,-1,np.int32)
            a[:len(tokens)-1]=tokens[:-1];b[:len(tokens)-1]=tokens[1:]
            qx.append(a);qy.append(b);meta.append({'document_id':doc_id,'occurrence':occurrence,'valid_tokens':len(tokens)-1})
        if len(meta)!=config[key]:raise ValueError('Insufficient distinct nonempty evaluation documents')
        np.save(root/f'{split}_x.npy',np.array(qx));np.save(root/f'{split}_y.npy',np.array(qy))
        write_json(root/f'{split}_provenance.json',meta);quality_meta[split]={'documents':len(meta),'valid_tokens':sum(r['valid_tokens'] for r in meta)}
    con.close()
    for p in (Path(config['data_dir']).expanduser()/'tokenizer').iterdir():
        if p.is_file():source_hashes[str(p.resolve())]=digest(p)
    files={p.name:digest(p) for p in root.iterdir() if p.is_file()}
    result={'schema':1,'split_salt':config['split_salt'],'source_files':source_hashes,'files':files,
        'document_occurrences':counts,'unique_documents':unique,'duplicate_occurrences':sum(counts.values())-unique,
        'exact_cross_split_overlap':overlap,'vocab_size':tok.get_vocab_size(),'quality':quality_meta,
        'training_batches':n_batches,'training_packing':'original BOS largest-fit / shortest-crop; provenance retained',
        'limitations':['Existing tokenizer training provenance incomplete; tokenizer may have seen evaluation text.',
            'Near-duplicate contamination not audited; exact document hashes only.',
            'Single local corpus source snapshot; no upstream revision provided.']}
    write_json(root/'manifest.json',result);return result


def verify(root,include_reserved=False):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if not include_reserved and (name.startswith('reserved_') or name=='documents.sqlite'):continue
        if digest(root/name)!=h:raise ValueError('Replay integrity mismatch: '+name)
    return manifest
