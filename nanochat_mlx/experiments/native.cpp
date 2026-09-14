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
static Bytes encode(const U*p,U rows,U cols,int mode){
    if(mode==0)return raw(p,rows,cols);
    std::unordered_map<U,U> dict; std::vector<U> vals,seq,rules,heights;
    for(U i=0;i<rows;++i){
        for(U j=0;j<cols;++j){
            U bits=p[size_t(i)*cols+j];
            if(bits==0)continue;
            // Preserve -0 and every nonfinite pattern with explicit RAW fallback.
            if(bits==0x80000000U||(bits&0x7f800000U)==0x7f800000U)return raw(p,rows,cols);
            auto entry=dict.emplace(bits,U(vals.size()));if(entry.second) vals.push_back(bits);
            Q symbol=1+Q(entry.first->second)*cols+j;
            if(symbol>=std::numeric_limits<U>::max())return raw(p,rows,cols);
            seq.push_back(U(symbol));
        }
        seq.push_back(0);
    }
    Q base64=1+Q(vals.size())*cols;
    if(base64>=std::numeric_limits<U>::max())return raw(p,rows,cols);
    U base=U(base64), maxheight=0;
    // Exact most-frequent adjacent-pair substitution; deterministic tie by symbol pair.
    // This frequency-scan implementation is compiled, but not the linear-time author implementation.
    while(true){
        std::unordered_map<Q,U> freq;
        for(size_t i=1;i<seq.size();++i)if(seq[i-1]&&seq[i])++freq[(Q(seq[i-1])<<32)|seq[i]];
        Q best=0;U count=1;
        for(auto &kv:freq)if(kv.second>count||(kv.second==count&&count>1&&kv.first<best)){best=kv.first;count=kv.second;}
        if(count<2)break;
        U a=U(best>>32),b=U(best), id=base+U(rules.size()/2);
        if(Q(base)+rules.size()/2>=std::numeric_limits<U>::max())return raw(p,rows,cols);
        rules.push_back(a);rules.push_back(b);
        U ht=1+std::max(a<base?0:heights[a-base],b<base?0:heights[b-base]);heights.push_back(ht);maxheight=std::max(maxheight,ht);
        size_t out=0;
        for(size_t i=0;i<seq.size();++i){if(i+1<seq.size()&&seq[i]==a&&seq[i+1]==b){seq[out++]=id;++i;}else seq[out++]=seq[i];}
        seq.resize(out);
    }
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
        // Dictionary and RePair construction were still performed and are timed.
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
