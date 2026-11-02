"""Download only the seven named public files and require the recorded SHA256 before saving."""
from pathlib import Path
from urllib.request import urlopen,Request
from zipfile import ZipFile
from io import BytesIO
import hashlib
root=Path(__file__).resolve().parents[1];raw=root/'data/raw';raw.mkdir(parents=True,exist_ok=True)
for line in (root/'data/SHA256SUMS.txt').read_text().splitlines():
 expected,name=line.split();dest=raw/name
 if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:print('Verified',name);continue
 if name.endswith('.xpt'):url='https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/'+name;data=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=180).read()
 else:
  year=next(y for y in ['19','22','25'] if 'nhis'+y in name);archive='Adult19csv.zip' if year=='19' else 'adult'+year+'csv.zip';url='https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/20'+year+'/'+archive
  with ZipFile(BytesIO(urlopen(url,timeout=180).read())) as z:
   found=[p for p in z.namelist() if Path(p).name.lower()=='adult'+year+'.csv']
   if len(found)!=1:raise RuntimeError('Ambiguous CSV in CDC archive')
   data=z.read(found[0])
 if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Hash mismatch; file not saved: '+name)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);print('Downloaded and verified',name)
