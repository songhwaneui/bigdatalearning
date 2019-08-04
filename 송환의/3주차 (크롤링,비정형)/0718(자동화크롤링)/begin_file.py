
# coding: utf-8

# In[ ]:


import re
from urllib import *
from html import unescape
import requests
import json 
headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36"}


# In[ ]:


from urllib import error
import requests
import time
import bs4
def download(method,url,param=None,data=None, timeout=1, maxretries=3):
    #res = requests.request(method,url,param=param,data=data,headers=headers)
    
    try:
        resp=requests.request(method, url,params=param,data=data, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if 500<=e.response.status_code<600 and maxretries>0:
            print(maxretries)
            time.sleep(timeout)
            download(method,url,param,data,timeout,maxretries-1)
        else:
            print(e.response.status_code)
            print(e.response.reason)
    return resp
    

