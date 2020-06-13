import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape
import glob
import sys,os
import imageio
import os.path as osp

# base  = '/home/asd/'
base  = sys.argv[1]

xml   = osp.join(base, 'xml/')
fpath = osp.join(base, 'forms/')
gpath = osp.join(base, 'pargs/')

forms = glob.glob(xml+'*.xml')
c=0
for i in range(len(forms)):
    print(forms[i])
    rt = ET.parse(forms[i]).getroot()
    xmin, ymin = [sys.maxsize] * 2
    xmax, ymax = [0] * 2
    ftxt = ''
    for line in rt.findall('.//line'):
        txt,id = line.attrib['text'],line.attrib['id']
        txt = unescape(txt, {"&apos;": "'", "&quot;": '"'})
        ftxt = ftxt + txt + '\n'

        for cmp in line.findall('.//cmp'):
            dm = list(map(int,[ cmp.attrib['x'],cmp.attrib['y'],cmp.attrib['width'],cmp.attrib['height'] ]))
            xmin = min(xmin,dm[0])
            ymin = min(ymin,dm[1])
            xmax = max(xmax,dm[0]+dm[2])
            ymax = max(ymax,dm[1]+dm[3])

    ftxt = ftxt[:-1]
    marg = 5
    xmin -= marg
    ymin -= marg
    xmax += marg
    ymax += marg

    frmid = os.path.splitext( os.path.basename(forms[i]) )[0]
    frmfl = fpath + frmid + '.png'
    frm = imageio.imread(frmfl)
    imageio.imsave(gpath+frmid+'.png',frm[ymin:ymax,xmin:xmax])
    open(gpath + frmid + '.txt','w').write(ftxt)
