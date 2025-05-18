import pyopencl as cl
import pyopencl._cl as _cl 
import numpy as np
import hashlib
import time
import argparse
import itertools 
import os 
import re 

KERNEL_MAX_STRING_LENGTH = 55

DEFAULT_MIN_LEN = 1
DEFAULT_CHARSET_STR = "abcdefghijklmnopqrstuvwxyz0123456789"
DEFAULT_ALGORITHM = "sha256"
DEFAULT_DEVICE_TYPE = "gpu"
DEFAULT_GEN_MODE = "charset" 

HASH_INFO = {
    "md5":    {"digest_size": 16, "hex_len": 32, "type_enum": 0, "hash_obj_name": "md5"},
    "sha1":   {"digest_size": 20, "hex_len": 40, "type_enum": 1, "hash_obj_name": "sha1"},
    "sha256": {"digest_size": 32, "hex_len": 64, "type_enum": 2, "hash_obj_name": "sha256"},
}

ctx = None
queue = None

kernel_code = """
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

// Common ROTATE_LEFT for MD5 and SHA1
#define ROTL(x, n) ((x << n) | (x >> (32 - n)))

// --- START MD5 Implementation ---
__constant uint md5_t[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};
#define MD5_F(x, y, z) ((x & y) | (~x & z))
#define MD5_G(x, y, z) ((x & z) | (y & ~z))
#define MD5_H(x, y, z) (x ^ y ^ z)
#define MD5_I(x, y, z) (y ^ (x | ~z))
void md5_transform(uint state[4], const uchar block[64]) {
    uint a = state[0], b = state[1], c = state[2], d = state[3]; uint m[16];
    for (int i = 0; i < 16; ++i) { m[i] = (uint)block[i*4+0] | (uint)block[i*4+1]<<8 | (uint)block[i*4+2]<<16 | (uint)block[i*4+3]<<24; }
    #define MD5_FF(a,b,c,d,k,s,i) a = b + (ROTL( (a + MD5_F(b,c,d) + m[k] + md5_t[i-1]), s))
    MD5_FF(a,b,c,d, 0, 7, 1); MD5_FF(d,a,b,c, 1,12, 2); MD5_FF(c,d,a,b, 2,17, 3); MD5_FF(b,c,d,a, 3,22, 4);
    MD5_FF(a,b,c,d, 4, 7, 5); MD5_FF(d,a,b,c, 5,12, 6); MD5_FF(c,d,a,b, 6,17, 7); MD5_FF(b,c,d,a, 7,22, 8);
    MD5_FF(a,b,c,d, 8, 7, 9); MD5_FF(d,a,b,c, 9,12,10); MD5_FF(c,d,a,b,10,17,11); MD5_FF(b,c,d,a,11,22,12);
    MD5_FF(a,b,c,d,12, 7,13); MD5_FF(d,a,b,c,13,12,14); MD5_FF(c,d,a,b,14,17,15); MD5_FF(b,c,d,a,15,22,16);
    #define MD5_GG(a,b,c,d,k,s,i) a = b + (ROTL( (a + MD5_G(b,c,d) + m[k] + md5_t[i-1]), s))
    MD5_GG(a,b,c,d, 1, 5,17); MD5_GG(d,a,b,c, 6, 9,18); MD5_GG(c,d,a,b,11,14,19); MD5_GG(b,c,d,a, 0,20,20);
    MD5_GG(a,b,c,d, 5, 5,21); MD5_GG(d,a,b,c,10, 9,22); MD5_GG(c,d,a,b,15,14,23); MD5_GG(b,c,d,a, 4,20,24);
    MD5_GG(a,b,c,d, 9, 5,25); MD5_GG(d,a,b,c,14, 9,26); MD5_GG(c,d,a,b, 3,14,27); MD5_GG(b,c,d,a, 8,20,28);
    MD5_GG(a,b,c,d,13, 5,29); MD5_GG(d,a,b,c, 2, 9,30); MD5_GG(c,d,a,b, 7,14,31); MD5_GG(b,c,d,a,12,20,32);
    #define MD5_HH(a,b,c,d,k,s,i) a = b + (ROTL( (a + MD5_H(b,c,d) + m[k] + md5_t[i-1]), s))
    MD5_HH(a,b,c,d, 5, 4,33); MD5_HH(d,a,b,c, 8,11,34); MD5_HH(c,d,a,b,11,16,35); MD5_HH(b,c,d,a,14,23,36);
    MD5_HH(a,b,c,d, 1, 4,37); MD5_HH(d,a,b,c, 4,11,38); MD5_HH(c,d,a,b, 7,16,39); MD5_HH(b,c,d,a,10,23,40);
    MD5_HH(a,b,c,d,13, 4,41); MD5_HH(d,a,b,c, 0,11,42); MD5_HH(c,d,a,b, 3,16,43); MD5_HH(b,c,d,a, 6,23,44);
    MD5_HH(a,b,c,d, 9, 4,45); MD5_HH(d,a,b,c,12,11,46); MD5_HH(c,d,a,b,15,16,47); MD5_HH(b,c,d,a, 2,23,48);
    #define MD5_II(a,b,c,d,k,s,i) a = b + (ROTL( (a + MD5_I(b,c,d) + m[k] + md5_t[i-1]), s))
    MD5_II(a,b,c,d, 0, 6,49); MD5_II(d,a,b,c, 7,10,50); MD5_II(c,d,a,b,14,15,51); MD5_II(b,c,d,a, 5,21,52);
    MD5_II(a,b,c,d,12, 6,53); MD5_II(d,a,b,c, 3,10,54); MD5_II(c,d,a,b,10,15,55); MD5_II(b,c,d,a, 1,21,56);
    MD5_II(a,b,c,d, 8, 6,57); MD5_II(d,a,b,c,15,10,58); MD5_II(c,d,a,b, 6,15,59); MD5_II(b,c,d,a,13,21,60);
    MD5_II(a,b,c,d, 4, 6,61); MD5_II(d,a,b,c,11,10,62); MD5_II(c,d,a,b, 2,15,63); MD5_II(b,c,d,a, 9,21,64);
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
}
void md5_process_short_message(const uchar* message, uchar output_hash[16]) {
    uint h_init[4]={0x67452301,0xefcdab89,0x98badcfe,0x10325476}; uint state[4];
    for(int i=0;i<4;++i)state[i]=h_init[i]; uchar block[64];
    for(int i=0;i<%(length)d;++i)block[i]=message[i]; block[%(length)d]=0x80;
    for(int i=%(length)d+1;i<56;++i)block[i]=0x00; ulong bit_len=(ulong)%(length)d*8;
    block[56]=(uchar)(bit_len); block[57]=(uchar)(bit_len>>8); block[58]=(uchar)(bit_len>>16); block[59]=(uchar)(bit_len>>24);
    block[60]=(uchar)(bit_len>>32); block[61]=(uchar)(bit_len>>40); block[62]=(uchar)(bit_len>>48); block[63]=(uchar)(bit_len>>56);
    md5_transform(state,block);
    for(int i=0;i<4;++i){ output_hash[i*4+0]=(uchar)(state[i]); output_hash[i*4+1]=(uchar)(state[i]>>8); output_hash[i*4+2]=(uchar)(state[i]>>16); output_hash[i*4+3]=(uchar)(state[i]>>24); }
}
// --- END MD5 Implementation ---
// --- START SHA1 Implementation ---
__constant uint sha1_k[4]={0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xCA62C1D6};
uint sha1_f(uint t,uint b,uint c,uint d){
    if(t<20)return(b&c)|(~b&d); if(t<40)return b^c^d; if(t<60)return(b&c)|(b&d)|(c&d); return b^c^d;
}
void sha1_transform(uint state[5],const uchar block[64]){
    uint w[80]; for(int i=0;i<16;++i){w[i]=(uint)block[i*4+0]<<24|(uint)block[i*4+1]<<16|(uint)block[i*4+2]<<8|(uint)block[i*4+3];}
    for(int i=16;i<80;++i){w[i]=ROTL(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);}
    uint a=state[0],b=state[1],c=state[2],d=state[3],e=state[4];
    for(int t=0;t<80;++t){uint temp=ROTL(a,5)+sha1_f(t,b,c,d)+e+w[t]+sha1_k[t/20];e=d;d=c;c=ROTL(b,30);b=a;a=temp;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;
}
void sha1_process_short_message(const uchar* message,uchar output_hash[20]){
    uint h_init[5]={0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0}; uint state[5];
    for(int i=0;i<5;++i)state[i]=h_init[i]; uchar block[64];
    for(int i=0;i<%(length)d;++i)block[i]=message[i]; block[%(length)d]=0x80;
    for(int i=%(length)d+1;i<56;++i)block[i]=0x00; ulong bit_len=(ulong)%(length)d*8;
    block[56]=(uchar)(bit_len>>56);block[57]=(uchar)(bit_len>>48);block[58]=(uchar)(bit_len>>40);block[59]=(uchar)(bit_len>>32);
    block[60]=(uchar)(bit_len>>24);block[61]=(uchar)(bit_len>>16);block[62]=(uchar)(bit_len>>8);block[63]=(uchar)(bit_len);
    sha1_transform(state,block);
    for(int i=0;i<5;++i){output_hash[i*4+0]=(uchar)(state[i]>>24);output_hash[i*4+1]=(uchar)(state[i]>>16);output_hash[i*4+2]=(uchar)(state[i]>>8);output_hash[i*4+3]=(uchar)(state[i]);}
}
// --- END SHA1 Implementation ---
// --- START SHA256 Implementation ---
__constant uint sha256_k[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define SHA256_ROTR(x,n)((x>>n)|(x<<(32-n)))
#define SHA256_SHR(x,n)(x>>n)
#define SHA256_Ch(x,y,z)((x&y)^(~x&z))
#define SHA256_Maj(x,y,z)((x&y)^(x&z)^(y&z))
#define SHA256_Sigma0(x)(SHA256_ROTR(x,2)^SHA256_ROTR(x,13)^SHA256_ROTR(x,22))
#define SHA256_Sigma1(x)(SHA256_ROTR(x,6)^SHA256_ROTR(x,11)^SHA256_ROTR(x,25))
#define SHA256_sigma0(x)(SHA256_ROTR(x,7)^SHA256_ROTR(x,18)^SHA256_SHR(x,3))
#define SHA256_sigma1(x)(SHA256_ROTR(x,17)^SHA256_ROTR(x,19)^SHA256_SHR(x,10))
void sha256_transform(uint state[8],const uchar block[64]){
    uint w[64];for(int i=0;i<16;++i){w[i]=(uint)block[i*4+0]<<24|(uint)block[i*4+1]<<16|(uint)block[i*4+2]<<8|(uint)block[i*4+3];}
    for(int i=16;i<64;++i){w[i]=SHA256_sigma1(w[i-2])+w[i-7]+SHA256_sigma0(w[i-15])+w[i-16];}
    uint a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;++i){uint T1=h+SHA256_Sigma1(e)+SHA256_Ch(e,f,g)+sha256_k[i]+w[i];uint T2=SHA256_Sigma0(a)+SHA256_Maj(a,b,c);h=g;g=f;f=e;e=d+T1;d=c;c=b;b=a;a=T1+T2;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}
void sha256_process_short_message(const uchar* message,uchar output_hash[32]){
    uint h_init[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x50c722f3,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint state[8];for(int i=0;i<8;++i)state[i]=h_init[i]; uchar block[64];
    for(int i=0;i<%(length)d;++i)block[i]=message[i]; block[%(length)d]=0x80;
    for(int i=%(length)d+1;i<56;++i)block[i]=0x00; ulong bit_len=(ulong)%(length)d*8;
    block[56]=(uchar)(bit_len>>56);block[57]=(uchar)(bit_len>>48);block[58]=(uchar)(bit_len>>40);block[59]=(uchar)(bit_len>>32);
    block[60]=(uchar)(bit_len>>24);block[61]=(uchar)(bit_len>>16);block[62]=(uchar)(bit_len>>8);block[63]=(uchar)(bit_len);
    sha256_transform(state,block);
    for(int i=0;i<8;++i){output_hash[i*4+0]=(uchar)(state[i]>>24);output_hash[i*4+1]=(uchar)(state[i]>>16);output_hash[i*4+2]=(uchar)(state[i]>>8);output_hash[i*4+3]=(uchar)(state[i]);}
}
// --- END SHA256 Implementation ---
// --- Main Brute-force Kernel ---
__constant uchar charset_values[%(base)d] = %(charset_data_str)s;
__kernel void brute_hash(
    const ulong current_kernel_offset, __global const uchar *target_hash_bytes,
    __global uint *found_flag, __global uchar *result_string,
    const ulong num_actual_candidates_in_batch, const int hash_type_selector,
    const int digest_size_actual
){
    ulong local_idx=get_global_id(0); if(local_idx>=num_actual_candidates_in_batch){return;}
    if(found_flag[0]==1){return;}
    ulong gid=local_idx+current_kernel_offset; uchar input_candidate[%(length)d]; ulong temp_val=gid;
    for(int i=%(length)d-1;i>=0;--i){input_candidate[i]=charset_values[temp_val %% %(base)d]; temp_val=temp_val/%(base)d;}
    uchar computed_hash[32]; 
    if(hash_type_selector==0){md5_process_short_message(input_candidate,computed_hash);}
    else if(hash_type_selector==1){sha1_process_short_message(input_candidate,computed_hash);}
    else{sha256_process_short_message(input_candidate,computed_hash);}
    int match=1; for(int i=0;i<digest_size_actual;++i){if(computed_hash[i]!=target_hash_bytes[i]){match=0;break;}}
    if(match){if(atomic_cmpxchg(found_flag,0,1)==0){for(int i=0;i<%(length)d;++i){result_string[i]=input_candidate[i];}}}
}
"""

def setup_opencl(preferred_device_type_str="any"):
    global ctx, queue
    if ctx is not None and queue is not None: return
    platforms = cl.get_platforms()
    if not platforms: raise RuntimeError("No OpenCL platforms found.")

    dev_type = cl.device_type.GPU if preferred_device_type_str.lower() == "gpu" else \
               cl.device_type.CPU if preferred_device_type_str.lower() == "cpu" else cl.device_type.ALL

    selected_device = None
    for p_idx, p in enumerate(platforms):
        try:
            devices = p.get_devices(device_type=dev_type)
            if devices:
                selected_device = devices[0]
                print(f"Found {p.get_info(cl.platform_info.NAME)} {selected_device.name}")
                break
        except cl.RuntimeError: continue

    if selected_device is None and dev_type != cl.device_type.ALL:
        print(f"Preferred device type '{preferred_device_type_str}' not found, trying any GPU...")
        for p_idx, p in enumerate(platforms):
            try:
                devices = p.get_devices(device_type=cl.device_type.GPU)
                if devices: selected_device = devices[0]; print(f"Found GPU: {selected_device.name}"); break
            except: continue
        if selected_device is None:
            print("No GPU found, trying any CPU...")
            for p_idx, p in enumerate(platforms):
                try:
                    devices = p.get_devices(device_type=cl.device_type.CPU)
                    if devices: selected_device = devices[0]; print(f"Found CPU: {selected_device.name}"); break
                except: continue

    if selected_device is None: raise RuntimeError("No OpenCL GPU or CPU devices found.")
    ctx = cl.Context([selected_device])
    queue = cl.CommandQueue(ctx)
    print(f"Using OpenCL device: {selected_device.name}")

def _extract_from_error_record(record_obj):
    """Helper to get meaningful info from a pyopencl._cl._ErrorRecord object."""
    if not isinstance(record_obj, _cl._ErrorRecord):
        return None

    routine = getattr(record_obj, 'routine', None)
    msg = getattr(record_obj, 'msg', None)

    if isinstance(msg, bytes): msg = msg.decode(errors='replace').strip()
    elif isinstance(msg, str): msg = msg.strip()

    if msg and msg != "None" and not msg.startswith("<") and not msg.endswith(">"): 
        if routine and routine != "None":
            return f"Routine: {routine}, Message: {msg}"
        return f"Message: {msg}"

    record_str = str(record_obj)
    if "<pyopencl._cl._ErrorRecord object at" in record_str :

        nested_what = getattr(record_obj, 'what', None)
        if isinstance(nested_what, (str, bytes)):
             nested_what_str = nested_what.decode(errors='replace') if isinstance(nested_what, bytes) else str(nested_what)
             if nested_what_str.strip() and "<pyopencl._cl._ErrorRecord object at" not in nested_what_str:
                 return f"Nested 'what' in _ErrorRecord: {nested_what_str.strip()}"
        return f"Uninformative _ErrorRecord object (routine='{routine}', msg='{msg}'). String repr: {record_str}"

    return record_str.strip() 

def _get_build_log(e: cl.RuntimeError):
    """Extracts detailed build logs from a PyOpenCL RuntimeError."""
    log_parts = []

    if hasattr(e, 'code') and e.code == cl.status_code.BUILD_PROGRAM_FAILURE: 

        if hasattr(e, 'stdout') and e.stdout:
            stdout_log = e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes) else str(e.stdout)
            log_parts.append(f"--- Build STDOUT ---\n{stdout_log.strip()}")
        if hasattr(e, 'stderr') and e.stderr:
            stderr_log = e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else str(e.stderr)
            log_parts.append(f"--- Build STDERR ---\n{stderr_log.strip()}")

        if hasattr(e, 'args') and len(e.args) >= 2 and isinstance(e.args[1], list):
            additional_arg_logs = []
            for item_idx, item in enumerate(e.args[1]):
                if isinstance(item, tuple) and len(item) == 2:
                    device, build_log_data = item
                    dev_name_str = getattr(device, 'name', f'Device {item_idx}')

                    current_log_str = None
                    if isinstance(build_log_data, _cl._ErrorRecord):
                        current_log_str = _extract_from_error_record(build_log_data)
                    elif isinstance(build_log_data, (str, bytes)):
                        current_log_str = build_log_data.decode(errors='replace') if isinstance(build_log_data, bytes) else str(build_log_data)

                    if current_log_str and current_log_str.strip() and "Uninformative _ErrorRecord object" not in current_log_str:
                        additional_arg_logs.append(f"--- Log for Device: {dev_name_str} (from e.args[1]) ---\n{current_log_str.strip()}")

            if additional_arg_logs:
                if log_parts: 
                    log_parts.append("\n--- Additional logs from e.args[1] for BuildError ---")
                log_parts.extend(additional_arg_logs)

        if log_parts: return "\n".join(log_parts)

    if hasattr(e, 'what') and callable(e.what):
        try:
            what_val = e.what()
            if isinstance(what_val, _cl._ErrorRecord):
                extracted = _extract_from_error_record(what_val)
                if extracted: log_parts.append(f"Log from e.what() (_ErrorRecord):\n{extracted}")
            elif isinstance(what_val, (str, bytes)):
                log_str = (what_val.decode(errors='replace') if isinstance(what_val, bytes) else str(what_val)).strip()
                if log_str: log_parts.append(f"Log from e.what():\n{log_str}")
        except Exception: pass

    if hasattr(e, 'what') and not (hasattr(e, 'what') and callable(e.what)): 
        what_attr = e.what
        if isinstance(what_attr, _cl._ErrorRecord):
            extracted = _extract_from_error_record(what_attr)
            if extracted: log_parts.append(f"Log from e.what attribute (_ErrorRecord):\n{extracted}")
        elif isinstance(what_attr, (str, bytes)):
            log_str = (what_attr.decode(errors='replace') if isinstance(what_attr, bytes) else str(what_attr)).strip()
            if log_str: log_parts.append(f"Log from e.what attribute:\n{log_str}")

    if hasattr(e, 'args') and e.args:
        for i, arg_item in enumerate(e.args):
            if isinstance(arg_item, _cl._ErrorRecord):
                extracted = _extract_from_error_record(arg_item)
                if extracted: log_parts.append(f"Log from e.args[{i}] (_ErrorRecord):\n{extracted}")
            elif isinstance(arg_item, (str, bytes)):
                log_str = (arg_item.decode(errors='replace') if isinstance(arg_item, bytes) else str(arg_item)).strip()

                if log_str and log_str not in str(e): 
                    log_parts.append(f"Log from e.args[{i}]:\n{log_str}")

    if log_parts:

        unique_logs = []
        seen_logs = set()
        for log_entry in log_parts:

            if "<pyopencl._cl._ErrorRecord object at" in log_entry and "Message:" not in log_entry and "Routine:" not in log_entry :
                continue
            if "Uninformative _ErrorRecord object" in log_entry and len(log_parts) > 1: 
                continue

            trimmed_log = log_entry.strip()
            if trimmed_log and trimmed_log not in seen_logs:
                unique_logs.append(trimmed_log)
                seen_logs.add(trimmed_log)
        if unique_logs:
            return "\n".join(unique_logs)

    return f"Could not extract detailed build log. Raw exception string: {str(e)}"

def brute_force_on_device(alg_name, target_hash, length, charset_bytes, batch_mult):
    global ctx, queue
    info = HASH_INFO[alg_name.lower()]
    digest_size, type_enum = info["digest_size"], info["type_enum"]

    if not (0 < length <= KERNEL_MAX_STRING_LENGTH): return None

    target_b = bytes.fromhex(target_hash)
    charset_init = "{" + ",".join(map(str, charset_bytes)) + "}"
    kernel_src_formatted = kernel_code % {"length": length, "base": len(charset_bytes), "charset_data_str": charset_init}

    prg = None 
    try:
        prg = cl.Program(ctx, kernel_src_formatted).build()
    except cl.RuntimeError as e:
        build_log_details = _get_build_log(e)
        print(f"OpenCL Kernel Compile Err (len {length}): Code {e.code if hasattr(e, 'code') else 'N/A'}\n--- Compiler Log Start ---\n{build_log_details}\n--- Compiler Log End ---")
        return None

    mf = cl.mem_flags
    target_buf = cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=np.frombuffer(target_b, dtype=np.uint8))
    found_h = np.array([0],dtype=np.uint32); found_d = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=found_h)
    res_h = np.zeros(length,dtype=np.uint8); res_d = cl.Buffer(ctx, mf.WRITE_ONLY, size=res_h.nbytes)

    local_val = min(256, ctx.devices[0].max_work_group_size)
    cand_launch = local_val * batch_mult
    max_combs = len(charset_bytes)**length
    processed = 0; start_t = time.time()

    print(f"GPU: Total for len {length}: {max_combs:,}. Batch: {cand_launch:,} (Local: {local_val})")

    for offset in range(0, max_combs, cand_launch):
        batch_c = min(cand_launch, max_combs - offset)
        if batch_c == 0: continue
        global_s = ((batch_c + local_val - 1)//local_val)*local_val

        try:

            if prg is None: 
                print(f"Error: OpenCL program object is None before launch (len {length}). This indicates an earlier unnoticed issue.")
                return None
            prg.brute_hash(queue, (global_s,), (local_val,), np.uint64(offset), target_buf,
                           found_d, res_d, np.uint64(batch_c), np.int32(type_enum), np.int32(digest_size)).wait()
        except cl.RuntimeError as e:
            launch_log_details = _get_build_log(e) 
            print(f"OpenCL Launch Err (len {length}): Code {e.code if hasattr(e, 'code') else 'N/A'}\n--- Error Details Start ---\n{launch_log_details}\n--- Error Details End ---")
            return None

        processed += batch_c
        cl.enqueue_copy(queue, found_h, found_d).wait()
        if found_h[0] == 1:
            cl.enqueue_copy(queue, res_h, res_d).wait()
            found_s = res_h.tobytes().decode(errors='ignore')
            elap = time.time()-start_t; hps = processed/elap if elap>0 else float('inf')
            print(f"\r[✓] GPU Match (len {length}): {found_s}. Attempts: {processed:,}. Time: {elap:.2f}s. HPS: {hps:,.0f} H/s" + " "*20)
            return found_s
        elap = time.time()-start_t
        if elap>0: hps=processed/elap; prog=processed/max_combs*100
        else: hps=float('inf'); prog=0
        print(f"\r[*] GPU Len {length}: {processed:,} ({prog:.2f}%) | {elap:.2f}s | {hps:,.0f} H/s", end="")
    print(f"\r[-] GPU: No match for len {length}." + " "*40)
    return False

def brute_force_on_cpu_for_length(alg_name, target_hash, length, charset_bytes, overall_start_t):
    info = HASH_INFO[alg_name.lower()]
    hashlib_name = info["hash_obj_name"]
    print(f"\n--- CPU: Len {length}, Charset: '{charset_bytes.decode(errors='ignore')}' ---")

    charset_c = [bytes([b]) for b in charset_bytes]
    num_combs = len(charset_bytes)**length
    print(f"CPU: Total for len {length}: {num_combs:,}. (Warning: may be very slow)")

    start_t_len = time.time(); attempts = 0; progress_int = 1000000

    for cand_tuple in itertools.product(charset_c, repeat=length):
        cand_b = b"".join(cand_tuple)
        curr_hash = hashlib.new(hashlib_name, cand_b).hexdigest()
        attempts += 1
        if curr_hash == target_hash.lower():
            elap_total = time.time()-overall_start_t; elap_len = time.time()-start_t_len
            hps_len = attempts/elap_len if elap_len > 0 else float('inf')
            print(f"\r[✓] CPU Match (len {length}): {cand_b.decode(errors='ignore')}" + " "*50)
            print(f"    Attempts (CPU len {length}): {attempts:,}. Time: {elap_len:.2f}s. HPS: {hps_len:,.0f} H/s")
            print(f"    Total script time: {elap_total:.2f}s")
            return cand_b.decode(errors='ignore')
        if attempts % progress_int == 0:
            elap_len_so_far = time.time()-start_t_len
            hps_so_far = attempts/elap_len_so_far if elap_len_so_far > 0 else float('inf')
            prog_p = (attempts/num_combs)*100 if num_combs > 0 else 0
            print(f"\r[*] CPU Len {length}: {attempts:,} ({prog_p:.2f}%) | {elap_len_so_far:.2f}s | {hps_so_far:,.0f} H/s", end="")
    print(f"\r[-] CPU: No match for len {length} after {attempts:,} attempts." + " "*50)
    return False

def apply_leetspeak(word):
    subs = {'a': '@', 'e': '3', 'o': '0', 'i': '1', 'l': '1', 's': '$', 't': '7'}
    variants = {word}
    leet_word = ""
    for char_orig in word: 
        char_lower = char_orig.lower()
        leet_char = subs.get(char_lower, char_orig) 
        leet_word += leet_char
    variants.add(leet_word)

    leet_preserve_case_word = "".join(subs.get(c.lower(), c) for c in word)
    variants.add(leet_preserve_case_word)

    if word.lower() not in variants : variants.add(word.lower()) 
    if word.upper() not in variants : variants.add(word.upper()) 
    if word.capitalize() not in variants : variants.add(word.capitalize()) 

    base_for_leet = word.lower()
    leet_lower = "".join(subs.get(c, c) for c in base_for_leet)
    variants.add(leet_lower)
    variants.add(leet_lower.capitalize())

    return list(variants)

def brute_force_with_rules_on_cpu(
    algorithm_name, target_hash_hex_str, wordlist_file, overall_start_time
):
    if not wordlist_file or not os.path.exists(wordlist_file):
        print("[!] Rules-based generation requires a valid wordlist. Wordlist not found or not specified.")
        return False

    algo_info = HASH_INFO[algorithm_name.lower()]
    hashlib_name = algo_info["hash_obj_name"]

    print(f"\n--- CPU: Applying mutation rules to wordlist: {wordlist_file} ---")

    suffixes = ["", "1", "12", "123", "!", "@", "#", "$", "2022", "2023", "2024", "01"]

    base_words_checked = 0
    mutations_tried = 0
    start_time_rules = time.time()
    progress_interval_base = 1000 

    try:
        with open(wordlist_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_idx, line in enumerate(f):
                base_word_orig = line.strip()
                if not base_word_orig: continue
                base_words_checked += 1

                if base_words_checked % progress_interval_base == 0:
                    elapsed_rules = time.time() - start_time_rules
                    hps_mut = mutations_tried / elapsed_rules if elapsed_rules > 0 else float('inf')
                    print(f"\r[*] Rules: Base words: {base_words_checked:,}. Mutations: {mutations_tried:,} ({hps_mut:,.0f} M/s). Time: {elapsed_rules:.2f}s", end="")

                base_candidates_casing = {
                    base_word_orig,
                    base_word_orig.lower(),
                    base_word_orig.upper(),
                    base_word_orig.capitalize()
                }

                all_base_candidates = set()
                for bc_case in base_candidates_casing:
                    all_base_candidates.update(apply_leetspeak(bc_case))

                for candidate_root in all_base_candidates:
                    for suffix in suffixes:
                        mutated_word = candidate_root + suffix
                        mutations_tried += 1

                        computed_hash = hashlib.new(hashlib_name, mutated_word.encode('utf-8', errors='ignore')).hexdigest()
                        if computed_hash == target_hash_hex_str.lower():
                            elapsed_total = time.time() - overall_start_time
                            elapsed_rules = time.time() - start_time_rules
                            print("\r" + " " * 120 + "\r", end="") 
                            print(f"[✓] CPU Rules: Match found: '{mutated_word}' (from base: '{base_word_orig}')")
                            print(f"    Base words processed: {base_words_checked:,}. Total mutations tried: {mutations_tried:,}")
                            print(f"    Time (Rules phase): {elapsed_rules:.2f}s")
                            print(f"    Total script time: {elapsed_total:.2f}s")
                            return True

    except Exception as e:
        print(f"\n[!] Error during rules-based generation: {e}")
        return False

    elapsed_rules = time.time() - start_time_rules
    print("\r" + " " * 120 + "\r", end="") 
    print(f"[-] CPU Rules: No match found after applying rules to {base_words_checked:,} base words ({mutations_tried:,} mutations). Time: {elapsed_rules:.2f}s")
    return False

def main_brute_forcer(
    algorithm_name, target_hash_to_find, charset_str,
    min_len=DEFAULT_MIN_LEN, user_max_len=None,
    batch_size_multiplier=8192, device_type_pref=DEFAULT_DEVICE_TYPE,
    wordlist_file=None, gen_mode=DEFAULT_GEN_MODE
):
    overall_start_time = time.time()
    algorithm_name = algorithm_name.lower()
    if algorithm_name not in HASH_INFO:
        print(f"Error: Unsupported algorithm '{algorithm_name}'."); return False

    algo_info = HASH_INFO[algorithm_name]
    expected_hex_len, hashlib_name = algo_info["hex_len"], algo_info["hash_obj_name"]

    if wordlist_file:
        print(f"\n--- Attempting direct wordlist match: {wordlist_file} (CPU) ---")
        checked_count = 0
        try:
            if not os.path.exists(wordlist_file):
                 print(f"[!] Wordlist file not found: {wordlist_file}.")
            else:
                with open(wordlist_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        word = line.strip(); checked_count = i + 1
                        if not word: continue
                        if checked_count % 100000 == 0: print(f"\r[*] Wordlist: Checked {checked_count:,} words...", end="")
                        if hashlib.new(hashlib_name, word.encode('utf-8','ignore')).hexdigest() == target_hash_to_find.lower():
                            el_time = time.time()-overall_start_time
                            print(f"\r[✓] Match via wordlist: '{word}'. Time: {el_time:.4f}s. Checked: {checked_count:,}" + " "*30)
                            return True
                print(f"\r[-] No match in wordlist '{wordlist_file}' ({checked_count:,} words)." + " "*30)
        except Exception as e: print(f"\n[!] Error reading wordlist {wordlist_file}: {e}.")

    if not isinstance(target_hash_to_find, str) or \
       not all(c in "0123456789abcdefABCDEF" for c in target_hash_to_find) or \
       len(target_hash_to_find) != expected_hex_len:
        print(f"Error: Target hash '{target_hash_to_find}' for {algorithm_name.upper()} is invalid. "
              f"Must be a {expected_hex_len}-char hex string."); return False

    if gen_mode == "rules":
        if not wordlist_file: 
            print("[!] Rules-based generation mode (--gen-mode rules) requires a wordlist (--wordlist). Skipping rules generation.")
        else:
            if brute_force_with_rules_on_cpu(algorithm_name, target_hash_to_find, wordlist_file, overall_start_time):
                return True
            print(f"\n--- No match found via rules-based generation. ---") 

            return False 

    elif gen_mode == "charset":
        charset_bytes = charset_str.encode('utf-8')
        if not charset_bytes: print("Error: Charset cannot be empty."); return False

        if min_len <= 0 or (user_max_len is not None and user_max_len == 0):
            print(f"--- Checking for empty string (length 0) for {algorithm_name.upper()} ---")
            if hashlib.new(hashlib_name, b"").hexdigest() == target_hash_to_find.lower():
                el_time = time.time()-overall_start_time
                print(f"[✓] Match: (empty string). Time: {el_time:.4f}s")
                return True
            if user_max_len == 0: print(f"--- No match for length 0. Max_len is 0, stopping."); return False

        print(f"\n--- Starting charset-based brute-force (Mode: '{gen_mode}') ---")
        current_len = max(1, min_len)
        opencl_init_successful = False 
        cpu_msg_shown = False

        while True:
            if user_max_len is not None and current_len > user_max_len:
                print(f"\n--- Reached user max_len ({user_max_len}) for charset generation. ---"); break

            result = False 

            if current_len <= KERNEL_MAX_STRING_LENGTH:
                if not opencl_init_successful and ctx is None: 
                    try:
                        print(f"--- Init OpenCL for lengths <= {KERNEL_MAX_STRING_LENGTH} ---")
                        setup_opencl(device_type_pref)
                        opencl_init_successful = True 
                    except RuntimeError as e:
                        print(f"OpenCL Setup Error: {e}. OpenCL will not be used.")
                        opencl_init_successful = False 

                if opencl_init_successful: 
                    gpu_result = brute_force_on_device(algorithm_name, target_hash_to_find, current_len, charset_bytes, batch_size_multiplier)
                    if gpu_result is None: 
                        print(f"--- Error during OpenCL execution for length {current_len}. This length will be skipped on GPU. Check logs. ---")

                        return False 
                    elif isinstance(gpu_result, str): 
                        return True

                    result = gpu_result 

            should_try_cpu = False
            if current_len > KERNEL_MAX_STRING_LENGTH:
                should_try_cpu = True
                if not cpu_msg_shown: 
                    print(f"\n--- Switching to CPU for lengths > {KERNEL_MAX_STRING_LENGTH} (slower) ---")
                    cpu_msg_shown = True
            elif not opencl_init_successful and current_len <= KERNEL_MAX_STRING_LENGTH : 
                should_try_cpu = True
                if not cpu_msg_shown:
                    print(f"\n--- OpenCL not available or failed. Using CPU for length {current_len} ---")
                    cpu_msg_shown = True

            if should_try_cpu:

                if result is False: 
                    cpu_result = brute_force_on_cpu_for_length(algorithm_name, target_hash_to_find, current_len, charset_bytes, overall_start_time)
                    if cpu_result is None: 
                        print(f"--- Error during CPU processing for length {current_len}. Stopping. ---")
                        return False
                    elif isinstance(cpu_result, str): 
                        return True
                    result = cpu_result 

            current_len += 1

            if user_max_len is None and current_len > KERNEL_MAX_STRING_LENGTH + 10 and (not opencl_init_successful or current_len > KERNEL_MAX_STRING_LENGTH):

                print(f"\n--- Reached very long CPU length ({current_len-1}) without --max-len. Stopping. ---")
                print(f"--- Specify --max-len for longer CPU attempts. ---")
                break

        print(f"\n--- No match found via charset generation. ---") 

    else: 
        print(f"Error: Unknown generation mode '{gen_mode}'.")

    total_el_time = time.time() - overall_start_time
    print(f"Total script execution time: {total_el_time:.2f}s")
    return False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"OpenCL/CPU Hash Brute-Forcer. Supports MD5, SHA1, SHA256.")
    parser.add_argument("-a", "--algo", type=str, default=DEFAULT_ALGORITHM, choices=HASH_INFO.keys(), help=f"Hash algorithm (default: {DEFAULT_ALGORITHM})")
    parser.add_argument("-H", "--hash", type=str, required=True, help="Target hash hex string")
    parser.add_argument("-w", "--wordlist", type=str, default=None, help="Path to a wordlist .txt file (CPU-based, tried first, required for 'rules' mode).")

    parser.add_argument("--gen-mode", type=str, default=DEFAULT_GEN_MODE, choices=["charset", "rules"],
                        help=f"Generation mode. 'charset' for GPU/CPU brute-force, 'rules' for CPU-based mutations on wordlist (default: {DEFAULT_GEN_MODE}).")

    charset_group = parser.add_argument_group('Charset Mode Options (used if --gen-mode charset)')
    charset_group.add_argument("-c", "--charset", type=str, default=DEFAULT_CHARSET_STR, help=f"Charset for 'charset' mode generation (default: '{DEFAULT_CHARSET_STR}')")
    charset_group.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN, help=f"Min string length for 'charset' mode (default: {DEFAULT_MIN_LEN})")
    charset_group.add_argument("--max-len", type=int, default=None, help=f"Max string length for 'charset' mode (default: GPU up to {KERNEL_MAX_STRING_LENGTH}, then CPU limited by soft cap).")
    charset_group.add_argument("--batch-mult", type=int, default=8192, help="GPU batch size multiplier for 'charset' mode (default: 8192)")
    charset_group.add_argument("--device-type", type=str, default=DEFAULT_DEVICE_TYPE, choices=["gpu", "cpu", "any"], help=f"Preferred OpenCL device for GPU part of 'charset' mode (default: {DEFAULT_DEVICE_TYPE})")

    args = parser.parse_args()

    if args.gen_mode == "rules" and not args.wordlist:
        parser.error("--gen-mode 'rules' requires --wordlist to be specified.")

    print(f"--- {args.algo.upper()} Brute-Force Tool ---")
    print(f"Target Hash: {args.hash}")
    if args.wordlist: print(f"Wordlist: '{args.wordlist}'")
    print(f"Generation Mode: '{args.gen_mode}'")

    if args.gen_mode == "charset":
        print(f"  Charset for generation: '{args.charset}'")
        max_len_disp = args.max_len if args.max_len is not None else f"OpenCL up to {KERNEL_MAX_STRING_LENGTH}, then CPU (soft-capped)"
        print(f"  Length Range for generation: {args.min_len} to {max_len_disp}")
        print(f"  Batch Multiplier (OpenCL): {args.batch_mult}")
        print(f"  Preferred OpenCL Device: {args.device_type}")

    success = main_brute_forcer(
        algorithm_name=args.algo,
        target_hash_to_find=args.hash,
        charset_str=args.charset,
        min_len=args.min_len,
        user_max_len=args.max_len,
        batch_size_multiplier=args.batch_mult,
        device_type_pref=args.device_type,
        wordlist_file=args.wordlist,
        gen_mode=args.gen_mode
    )

    if success:
        print("\nBrute-force successful.")
    else:
        print("\nBrute-force finished: no match found or an error occurred.")
