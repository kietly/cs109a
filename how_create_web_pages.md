1 . `jupyter nbconvert notebooks/twitter-nlp.ipynb --template _support/markdown.tpl --output-dir . --to markdown`
2. Add this to the top of the generate md file.
---
title: NLP
notebook: twitter-nlp.ipynb
nav_include: 2
---

3. If you don't like the auto-gen TOC, remove them.
     
4. The nav_include # is the order of tabs on the web page. For example, Home = 0, Statement = 1, twitter-nlp= 2, twitter-eda = 3, and so on. 
6. Commit and push everything and go to the website in a bit to check the web site update.