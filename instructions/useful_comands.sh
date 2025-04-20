# print the project tree
tree -I '__pycache__|test|Instructions|assets|intelli.egg-info|build|dist|instructions'

# print python files
find . -name "*.py" -type f | while read file; do echo -e "\n===== $file ====="; cat "$file"; done