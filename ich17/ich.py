import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape
import glob
import sys,os
import scipy.misc
from PIL import Image
import os.path as osp
import unicodedata

base  = sys.argv[1]

xml    = osp.join(base,'page/')
pages  = glob.glob(xml+'*.xml')

rm = ["§","æ","­","|","‰","#","+","[","]","œ","̃","‒","*","□","°","†","‹","›","ο","—","£","τ","ν","‡","ψ","ι","α","κ","ω","η","℔","	","χ","ξ","₤","ε","π","~","μ","¬","Ζ","λ","Τ","Γ","І","̸","∫","·",">","♂","✓","Œ","♀","$","∆","ø","ρ","∇"]
print(len(pages))
a=[]
for i in range(len(pages)):
    rt = ET.parse(pages[i]).getroot()
    un = rt.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Unicode")
    txt = unescape(un.text, {"&apos;": "'", "&quot;": '"'})
    txt = unicodedata.normalize('NFKD', txt)
    txt = txt.translate({ord(i): None for i in rm})
    
    a.append(txt)
    open(osp.join(base,pages[i][-10:-4]+'.txt'),'w+').write(txt.strip())