// Independent adaptation of Ferragina et al., arXiv:2203.14540v2 §§2–4.
// No upstream source copied. C++17, strict FP (no fast-math / FMA contraction).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <limits>
#include <queue>
using U=uint32_t; using Q=uint64_t; using Bytes=std::vector<uint8_t>;
struct Header { U magic,version,rows,cols,nv,nr,nc,base,width,raw,height,reserved[5]; };
static_assert(sizeof(Header)==64);
static const U MAGIC=0x31474e4e;
static size_t packed_size(size_t n,U w){return ((n*w+63)/64)*8;}
static void put(Bytes &b,size_t off,size_t i,U w,U v){
    if(w==32){memcpy(b.data()+off+i*4,&v,4);return;}
    size_t bit=i*w,word=bit/64;U shift=U(bit%64);Q lo;
    memcpy(&lo,b.data()+off+word*8,8);lo|=Q(v)<<shift;memcpy(b.data()+off+word*8,&lo,8);
    if(shift+w>64){Q hi;memcpy(&hi,b.data()+off+(word+1)*8,8);hi|=Q(v)>>(64-shift);memcpy(b.data()+off+(word+1)*8,&hi,8);}
}
static U get(const uint8_t*b,size_t off,size_t i,U w){
    if(w==32){U v;memcpy(&v,b+off+i*4,4);return v;}
    size_t bit=i*w,word=bit/64;U shift=U(bit%64);Q lo;memcpy(&lo,b+off+word*8,8);
    Q v=lo>>shift;
    if(shift+w>64){Q hi;memcpy(&hi,b+off+(word+1)*8,8);v|=hi<<(64-shift);}
    return U(v&((Q(1)<<w)-1));
}
static U width(U x){U w=1;while(x>>=1)++w;return w;}
struct View{
    const uint8_t*b; Header h; size_t ro,co;
    View(const uint8_t *p,size_t size):b(p){
        if(size<64)throw std::runtime_error("short header");memcpy(&h,p,64);
        if(h.magic!=MAGIC||h.version!=1||h.width<1||h.width>32)throw std::runtime_error("invalid header");
        ro=64+size_t(h.nv)*4;co=ro+packed_size(size_t(h.nr)*2,h.width);
        size_t expected=h.raw?64+size_t(h.rows)*h.cols*4:co+packed_size(h.nc,h.width);
        if(expected!=size)throw std::runtime_error("invalid length");
        if(!h.raw && Q(h.base)!=1+Q(h.nv)*h.cols)throw std::runtime_error("terminal boundary");
    }
    U r(size_t i)const{return get(b,ro,i,h.width);}
    U c(size_t i)const{return get(b,co,i,h.width);}
    U bits(U i)const{U v;memcpy(&v,b+64+size_t(i)*4,4);return v;}
    float value(U i)const{U u=bits(i);float v;memcpy(&v,&u,4);return v;}
};
static Bytes raw(const U*p,U rows,U cols){
    Header h{};h.magic=MAGIC;h.version=1;h.rows=rows;h.cols=cols;h.raw=1;h.width=32;
    Bytes b(64+size_t(rows)*cols*4);memcpy(b.data(),&h,64);memcpy(b.data()+64,p,b.size()-64);return b;
}
// Independent incremental RePair: live linked positions, occurrence lists and
// a lazy frequency heap. Preserve the v1 pair tie-break and left-to-right order.
// No author source is incorporated; the wire format and FP32 evaluator stay v1.
static void repair(std::vector<U>&seq,U base,std::vector<U>&rules,U&maxheight){
    struct Pair { size_t count=0; std::vector<size_t> positions; };
    struct Candidate {
        size_t count; Q key;
        bool operator<(const Candidate&other)const{
            return count!=other.count?count<other.count:key>other.key;
        }
    };
    const size_t end=seq.size();
    // Retain the inexpensive v1 frequency pass for inputs with no repeated
    // pair. Only allocate linked positions and occurrence state when useful.
    std::unordered_map<Q,U> initial;
    bool repeated=false;
    for(size_t i=1;i<end;++i)if(seq[i-1]&&seq[i]){
        if(++initial[(Q(seq[i-1])<<32)|seq[i]]==2)repeated=true;
    }
    if(!repeated)return;
    std::vector<size_t> prev(end),next(end);
    std::unordered_map<Q,Pair> pairs;
    std::priority_queue<Candidate> heap;
    std::vector<U> heights;
    auto key=[&](size_t i){return (Q(seq[i])<<32)|seq[next[i]];};
    auto edge=[&](size_t i){return i<end&&next[i]<end&&seq[i]&&seq[next[i]];};
    for(size_t i=0;i<end;++i){prev[i]=i?i-1:end;next[i]=i+1;}
    for(const auto&kv:initial){
        pairs[kv.first].count=kv.second;
        if(kv.second>1)heap.push({kv.second,kv.first});
    }
    initial.clear();initial.rehash(0);
    // Existing pairs can only lose occurrences; every new pair contains the
    // newly assigned rule ID. Initial singleton occurrence lists are unnecessary.
    for(size_t i=0;i<end;++i)if(edge(i)){
        auto&p=pairs.at(key(i));if(p.count>1)p.positions.push_back(i);
    }
    auto remove=[&](size_t i){
        if(!edge(i))return;
        Q k=key(i);auto it=pairs.find(k);
        if(it==pairs.end()||!it->second.count)throw std::runtime_error("pair accounting");
        if(--it->second.count==0)pairs.erase(it);
        else if(it->second.count>1)heap.push({it->second.count,k});
    };
    auto add=[&](size_t i){
        if(!edge(i))return;
        Q k=key(i);auto&p=pairs[k];++p.count;p.positions.push_back(i);
        if(p.count>1)heap.push({p.count,k});
    };
    while(!heap.empty()){
        auto best=heap.top();heap.pop();auto it=pairs.find(best.key);
        if(it==pairs.end()||it->second.count!=best.count)continue;
        if(Q(base)+rules.size()/2>=std::numeric_limits<U>::max())throw std::overflow_error("rule ID");
        U a=U(best.key>>32),b=U(best.key),id=base+U(rules.size()/2);
        rules.push_back(a);rules.push_back(b);
        U ht=1+std::max(a<base?0:heights[a-base],b<base?0:heights[b-base]);
        heights.push_back(ht);maxheight=std::max(maxheight,ht);
        auto positions=std::move(it->second.positions);
        std::sort(positions.begin(),positions.end());
        for(size_t i:positions){
            if(!edge(i)||key(i)!=best.key)continue;
            size_t j=next[i],left=prev[i],right=next[j];
            remove(left);remove(i);remove(j);
            seq[i]=id;seq[j]=0;next[i]=right;next[j]=end;
            if(right<end)prev[right]=i;
            add(left);add(i);
        }
    }
    size_t out=0;
    for(size_t i=0;i<end;i=next[i])seq[out++]=seq[i];
    seq.resize(out);
}
static Bytes encode(const U*p,U rows,U cols,int mode){
    if(mode==0)return raw(p,rows,cols);
    size_t cutoff=std::numeric_limits<size_t>::max();
    if(mode==-1){
        const size_t cells=size_t(rows)*cols;
        // With d distinct values in at most N cells, at least max(0,2d-N)
        // values occur once. Their terminals cannot belong to any repeated
        // pair/rule, so each must remain in C. This lower bound also includes
        // the dictionary and header, but omits all other symbols and rules.
        auto impossible=[&](size_t d){
            Q terminal_base=1+Q(d)*cols;
            if(terminal_base>=std::numeric_limits<U>::max())return true;
            size_t singles=d>cells/2?2*d-cells:0;
            return 64+d*4+packed_size(singles,width(U(terminal_base)))>.95*(cells*4);
        };
        // The bound is monotone in d. Compute its cutoff once per block;
        // construction then needs only a cheap distinct-count comparison.
        size_t lo=0,hi=cells;
        while(lo<hi){size_t mid=lo+(hi-lo)/2;if(impossible(mid))hi=mid;else lo=mid+1;}
        cutoff=lo;
        if(cutoff==0)return raw(p,rows,cols);
    }
    std::unordered_map<U,U> dict; std::vector<U> vals,seq,rules;
    for(U i=0;i<rows;++i){
        for(U j=0;j<cols;++j){
            U bits=p[size_t(i)*cols+j];
            if(bits==0)continue;
            // Preserve -0 and every nonfinite pattern with explicit RAW fallback.
            if(bits==0x80000000U||(bits&0x7f800000U)==0x7f800000U)return raw(p,rows,cols);
            auto entry=dict.emplace(bits,U(vals.size()));if(entry.second) vals.push_back(bits);
            if(vals.size()>=cutoff)return raw(p,rows,cols);
            Q symbol=1+Q(entry.first->second)*cols+j;
            if(symbol>=std::numeric_limits<U>::max())return raw(p,rows,cols);
            seq.push_back(U(symbol));
        }
        seq.push_back(0);
    }
    Q base64=1+Q(vals.size())*cols;
    if(base64>=std::numeric_limits<U>::max())return raw(p,rows,cols);
    U base=U(base64), maxheight=0;
    try{repair(seq,base,rules,maxheight);}catch(const std::overflow_error&){return raw(p,rows,cols);}
    U maxid=base+(rules.empty()?0:U(rules.size()/2)-1),w=width(maxid);
    auto build=[&](U bits){
        Header h{};h.magic=MAGIC;h.version=1;h.rows=rows;h.cols=cols;h.nv=U(vals.size());h.nr=U(rules.size()/2);h.nc=U(seq.size());h.base=base;h.width=bits;h.height=maxheight;
        size_t ro=64+vals.size()*4,co=ro+packed_size(rules.size(),bits);
        Bytes b(co+packed_size(seq.size(),bits));memcpy(b.data(),&h,64);memcpy(b.data()+64,vals.data(),vals.size()*4);
        for(size_t i=0;i<rules.size();++i)put(b,ro,i,bits,rules[i]);
        for(size_t i=0;i<seq.size();++i)put(b,co,i,bits,seq[i]);
        return b;
    };
    if(mode==-1){
        size_t fixed=64+vals.size()*4+packed_size(rules.size(),32)+packed_size(seq.size(),32);
        size_t packed=64+vals.size()*4+packed_size(rules.size(),w)+packed_size(seq.size(),w);
        // Exact physical sizes decide representation before writing unused streams.
        // Blocks not rejected by the dictionary bound reach this final decision.
        if(std::min(fixed,packed)>.95*(size_t(rows)*cols*4))return raw(p,rows,cols);
        return build(fixed<=packed?32:w);
    }
    return build(mode==32?32:w);
}
extern "C" {
void* ng_encode(const U*p,U rows,U cols,int mode,size_t*size){
    try{auto*b=new Bytes(encode(p,rows,cols,mode));*size=b->size();return b;}catch(...){*size=0;return nullptr;}
}
const uint8_t* ng_data(void*p){return static_cast<Bytes*>(p)->data();}
void ng_free(void*p){delete static_cast<Bytes*>(p);}
int ng_decode(const uint8_t*b,size_t size,U*out){
    try{
        View v(b,size);auto h=v.h;
        if(h.raw){memcpy(out,b+64,size-64);return 0;}
        std::fill(out,out+size_t(h.rows)*h.cols,U(0));
        for(U i=0;i<h.nr;++i)for(U k=0;k<2;++k){U s=v.r(size_t(i)*2+k);if(!s||s>=Q(h.base)+i)throw std::runtime_error("cycle or delimiter rule");}
        U row=0,last=0;bool have=false;std::vector<U> stack;size_t expanded=0;
        for(U i=0;i<h.nc;++i){
            U s=v.c(i);if(s==0){if(row>=h.rows)throw std::runtime_error("rows");++row;have=false;continue;}
            stack.push_back(s);
            while(!stack.empty()){
                U t=stack.back();stack.pop_back();
                if(t>=h.base){U k=t-h.base;if(k>=h.nr)throw std::runtime_error("symbol");stack.push_back(v.r(size_t(k)*2+1));stack.push_back(v.r(size_t(k)*2));}
                else{
                    if(!t||!h.cols||row>=h.rows)throw std::runtime_error("terminal");
                    U col=(t-1)%h.cols,vi=(t-1)/h.cols;
                    if(vi>=h.nv||(have&&col<=last)||++expanded>size_t(h.rows)*h.cols)throw std::runtime_error("expansion");
                    out[size_t(row)*h.cols+col]=v.bits(vi);last=col;have=true;
                }
            }
        }
        if(row!=h.rows||(!h.nc&&h.rows)|| (h.nc&&v.c(h.nc-1)!=0))throw std::runtime_error("unterminated rows");
        return 0;
    }catch(...){return -1;}
}
int ng_matvec(const uint8_t*b,size_t size,const float*x,float*y,int transpose){
    try{
        View v(b,size);auto h=v.h;
        if(h.raw){const float *m=reinterpret_cast<const float*>(b+64);
            std::fill(y,y+(transpose?h.cols:h.rows),0.f);
            for(U i=0;i<h.rows;++i)for(U j=0;j<h.cols;++j){if(transpose)y[j]+=m[size_t(i)*h.cols+j]*x[i];else y[i]+=m[size_t(i)*h.cols+j]*x[j];}return 0;}
        std::vector<float> q(h.nr,0.f);
        auto eval=[&](U s)->float{if(s>=h.base)return q[s-h.base];return v.value((s-1)/h.cols)*x[(s-1)%h.cols];};
        if(!transpose){
            for(U k=0;k<h.nr;++k)q[k]=eval(v.r(size_t(k)*2))+eval(v.r(size_t(k)*2+1));
            U row=0;float sum=0;for(U i=0;i<h.nc;++i){U s=v.c(i);if(!s){y[row++]=sum;sum=0;}else sum+=eval(s);}
        }else{
            std::fill(y,y+h.cols,0.f);
            auto add=[&](U s,float a){if(s>=h.base)q[s-h.base]+=a;else y[(s-1)%h.cols]+=v.value((s-1)/h.cols)*a;};
            U row=0;for(U i=0;i<h.nc;++i){U s=v.c(i);if(!s)++row;else add(s,x[row]);}
            for(size_t k=h.nr;k-->0;){add(v.r(k*2),q[k]);add(v.r(k*2+1),q[k]);}
        }
        return 0;
    }catch(...){return -1;}
}
}
