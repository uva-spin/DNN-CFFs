cd Github/DNN-CFFs
git config --global user.name "mfarooq786"
git config --global user.email "2714befarooq@gmail.com"
ssh-keygen -t ed25519 -C "2714befarooq@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
git remote set-url origin git@github.com:uva-spin/DNN-CFFs.git
git status
git add . 
git add —a
git commit -m "change"
git push origin main


