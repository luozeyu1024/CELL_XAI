"""Inventory and download complete GEO supplementary expression resources."""
import argparse
import concurrent.futures
import gzip
import json
from pathlib import Path
import re
import requests
import time


def fetch(url, path, chunk_mb=4, range_workers=2):
    """Stream a complete resource, with a validated range fallback."""
    part = path.with_name(path.name+'.part')
    if not path.exists() and not part.exists():
        try:
            with requests.get(url,headers={'Accept-Encoding':'identity'},stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                if response.status_code!=200:
                    raise ValueError('Expected a complete file response')
                length=response.headers.get('Content-Length')
                if length is None:
                    raise ValueError('Complete response has no declared byte length')
                total=int(length)
                if total<=0:
                    raise ValueError('Expected a nonempty file')
                with part.open('xb') as handle:
                    for block in response.iter_content(chunk_size=1024*1024):
                        handle.write(block)
                if part.stat().st_size!=total:
                    raise ValueError('Incomplete streaming download')
                part.rename(path)
                print('DONE',path,flush=True)
                return
        except (requests.RequestException,ValueError) as error:
            print('FALLBACK',path.name,str(error),flush=True)
    fetch_ranges(url,path,chunk_mb,range_workers)


def fetch_ranges(url, path, chunk_mb, range_workers):
    """Resume a verified prefix from byte ranges."""
    info = requests.get(url+'?range=0-0',headers={'Range':'bytes=0-0'},timeout=(30,90))
    info.raise_for_status()
    if info.status_code!=206:
        raise ValueError('Expected byte-range support: '+url)
    total = int(info.headers['Content-Range'].split('/')[-1])
    if path.exists():
        if path.stat().st_size!=total:
            raise ValueError('Existing resource has unexpected length: '+str(path))
        print('EXISTS',path,flush=True)
        return
    part = path.with_name(path.name+'.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset>total:
        raise ValueError('Partial resource exceeds expected length')
    def block(bounds):
        lo,hi = bounds
        for retry in range(10):
            try:
                r=requests.get(url+f'?range={lo}-{hi}',headers={'Range':f'bytes={lo}-{hi}'},timeout=(30,90))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{total}' or len(r.content)!=hi-lo+1:
                    raise ValueError('Byte range mismatch')
                return r.content
            except (requests.RequestException,ValueError) as error:
                print('RETRY',path.name,lo,retry+1,str(error)[:90],flush=True)
                if retry==9:
                    raise
                time.sleep(min(2**retry,30))
    with part.open('ab') as handle, concurrent.futures.ThreadPoolExecutor(max_workers=range_workers) as pool:
        while offset<total:
            step=chunk_mb*1024*1024
            spans=[(n,min(n+step-1,total-1)) for n in range(offset,min(offset+step*range_workers,total),step)]
            for value in pool.map(block,spans):
                handle.write(value)
                handle.flush()
                offset+=len(value)
            print('PROGRESS',path.name,offset,total,flush=True)
    part.rename(path)
    print('DONE',path,flush=True)

