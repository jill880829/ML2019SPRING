# Final Project: Intent Retrieval from Online News

## 1. Python Version and Toolkit Version

+   python version: **3.6.8**
+   toolkit version:
    +   numpy==**1.16.4**
    +   pandas==**0.24.2**
    +   jieba==**0.39**
    +   Whoosh==**2.7.4**
    +   scikit-learn==**0.21.2**
    +   gensim==**3.7.3**

---

## 2. Code Execution

For first running, type `bash retrieval.sh {output_path}` on terminal to execute. (It will download `url2content.json` and create the index directory for search)  
For example, the following command will output "result.csv" in current directory.

```sh
bash retrieval.sh ./result.csv
```

For further running, type `python ./src/jieba_whoosh.py {output_path}` on terminal to execute. (It simpily search from the precreated index directory)  
For example, the following command will output "result.csv" in current directory.

```sh
python ./src/jieba_whoosh.py ./result.csv
```


---

## 3. Note

When executing `retrieval.sh`, a folder named `indexdir` will be created in the current directory. If the execution of `retrieval.sh` is interrupted during the execution time for some reason(for example KeyboardInterrupt), you **must delete** the `indexdir` folder and run again.  
When executing `./src/jieba_whoosh.py`, `indexdir` **does not need** to be deleted, it provides quick searching.
