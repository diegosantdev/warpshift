import os
import shutil
from pipeline import DATA_DIR

repo_cache = os.path.join(DATA_DIR, 'repo-cache')
if os.path.exists(repo_cache):
    shutil.rmtree(repo_cache)
    print('Wiped repo-cache')
else:
    print('No repo-cache found')
