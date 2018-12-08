1 . `jupyter nbconvert notebooks/twitter-nlp.ipynb --template _support/markdown.tpl --output-dir . --to markdown`
3. Commit and push and go to the website in a bit to see the changes we have got. Its much better already.
4. Add **YAML** preambles and some TOC frontmatter to our markdown files. For example,
   
   a. `python _support/nbmd.py twitter-nlp.md` # run only once
    
5. Edit the markdown files to add a YAML tag `nav_include: <position #>`  respectively to the above markdown files. 
The position # is order of tabs on the web page. For example, Home = 0, Statement = 1, twitter-nlp= 2, twitter-eda = 3, and so on. 
6. Commit and push everything and go to the website in a bit to check the update.