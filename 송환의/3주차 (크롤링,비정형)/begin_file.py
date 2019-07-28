{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "from urllib import *\n",
    "from html import unescape\n",
    "import requests\n",
    "import json \n",
    "headers={\"user-agent\":\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from urllib import error\n",
    "import requests\n",
    "import time\n",
    "import bs4\n",
    "def download(method,url,param=None,data=None, timeout=1, maxretries=3):\n",
    "    #res = requests.request(method,url,param=param,data=data,headers=headers)\n",
    "    \n",
    "    try:\n",
    "        resp=requests.request(method, url,params=param,data=data, headers=headers)\n",
    "        resp.raise_for_status()\n",
    "    except requests.exceptions.HTTPError as e:\n",
    "        if 500<=e.response.status_code<600 and maxretries>0:\n",
    "            print(maxretries)\n",
    "            time.sleep(timeout)\n",
    "            download(method,url,param,data,timeout,maxretries-1)\n",
    "        else:\n",
    "            print(e.response.status_code)\n",
    "            print(e.response.reason)\n",
    "    return resp\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
