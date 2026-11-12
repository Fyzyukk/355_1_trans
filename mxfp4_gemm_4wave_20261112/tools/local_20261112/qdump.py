import json,glob,sys
d,qi=sys.argv[1],int(sys.argv[2])
c=json.load(open(glob.glob(f'{d}/att/ui_output*/code.json')[0]))['code']
mx=max(r[6] for r in c if r[0] and 'mfma' in r[0]); L=[r for r in c if r[0] and r[6]==mx]
nm=0
for r in L:
    op=r[0].split()[0]
    if nm//32==qi and 'mfma' not in op or ('mfma' in op and nm//32==qi and (r[8]/mx>3)):
        print(f'{nm%32:3d} lat {r[7]/mx:5.1f} stall {r[8]/mx:5.1f}  {r[0][:95]}')
    if 'mfma' in op: nm+=1
