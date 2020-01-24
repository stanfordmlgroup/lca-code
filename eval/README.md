# aihc-fall18-evaluator
Model Evaluators for the Fall 2018 Projects of AIHC Bootcamp (CBIR, CXR, and Payment)


### Usage

1. **Remove eval directory (if one exists)**
   - Run `rm -r eval` from your project directory and then `git rm --cached eval`.
2. **Add as submodule to project repository**
   - Simply cd into project repository and run `git submodule add https://github.com/stanfordmlgroup/aihc-fall18-evaluator.git ./eval`
   - Next, run `git submodule init` and `git submodule update`
   - You should see the evaluator files present in the eval folder.

3. **Updating Submodule**
   - When the evaluator repo is updated, the submodules themselves also have to be updated to the latest version in each project repo.
   - The submodule will not automatically update! So make sure you have the latest version of the evaluators.
   - To update a submodule for a specific project:
      - run `git submodule update --remote`
   - Another developer of the project can run the following to pull the changes:
      - run `git pull` to pull the changes for the project
      - run `git submodule update --remote` to update the submodule
      

