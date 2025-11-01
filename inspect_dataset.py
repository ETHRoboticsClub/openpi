from itertools import islice
from huggingface_hub import list_repo_tree

repo_id = "ETHRC/piper_towel_v0_with_rewards"
tree = list_repo_tree(repo_id, repo_type="dataset", recursive=True)
for entry in islice(tree, 40):
    print(entry.path)
