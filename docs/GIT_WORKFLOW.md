# Git Workflow - PhobiaShield

## Branch Strategy

We follow a simplified Git workflow optimized for a 14-day sprint with 3 team members.

---

## Branch Structure

```
main (production)
  ↑
develop (integration)
  ↑
feature/* (individual work)
```

### Branch Descriptions

- **`main`**: Production-ready code only
  - Protected branch
  - Requires pull request + review
  - Only merge from `develop`

- **`develop`**: Integration branch
  - Latest working features
  - Merge feature branches here
  - Test before merging to main

- **`feature/*`**: Individual feature branches
  - Format: `feature/member-task-name`
  - Examples:
    - `feature/gabriele-fpn-architecture`
    - `feature/memberA-dataset-preprocessing`
    - `feature/memberC-demo-interface`

---

## Workflow

### 1. Starting New Work

```bash
# Update develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-name-task-name

# Example: Gabriele working on FPN
git checkout -b feature/gabriele-fpn-loss-function
```

### 2. Working on Feature

```bash
# Make changes
# ... edit files ...

# Stage and commit
git add .
git commit -m "Add Focal Loss implementation"

# Push to remote
git push origin feature/gabriele-fpn-loss-function
```

### 3. Merging Feature

**Option A: Direct Merge (Small Changes)**

```bash
# Update develop
git checkout develop
git pull origin develop

# Merge feature
git merge feature/gabriele-fpn-loss-function

# Push
git push origin develop

# Delete feature branch
git branch -d feature/gabriele-fpn-loss-function
git push origin --delete feature/gabriele-fpn-loss-function
```

**Option B: Pull Request (Recommended for Major Changes)**

1. Push feature branch to GitHub
2. Open Pull Request on GitHub
3. Request review from team
4. Merge after approval
5. Delete feature branch

---

## Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only
- **style**: Code formatting (no logic change)
- **refactor**: Code restructuring (no feature change)
- **test**: Adding tests
- **chore**: Maintenance tasks

### Examples

**Good commits:**
```bash
git commit -m "feat(fpn): Add multi-scale detection heads"
git commit -m "fix(dataset): Correct class ID mapping for needle"
git commit -m "docs(readme): Update installation instructions"
```

**Bad commits:**
```bash
git commit -m "update"
git commit -m "fix stuff"
git commit -m "asdf"
```

### Scope Guidelines

- **model**: Neural network architecture
- **loss**: Loss function
- **dataset**: Data loading/processing
- **training**: Training loop/optimization
- **eval**: Evaluation/metrics
- **demo**: Demo interface
- **docs**: Documentation

---

## Collaboration Rules

### Before Starting Work

1. **Check existing branches**: `git branch -a`
2. **Coordinate with team**: Avoid duplicate work
3. **Pull latest develop**: Always start from updated code

### During Work

1. **Commit frequently**: Small, focused commits
2. **Push daily**: Backup your work
3. **Sync with develop**: Merge develop into feature if needed

```bash
# In your feature branch
git fetch origin
git merge origin/develop
```

### Resolving Conflicts

```bash
# If merge conflict occurs
git status  # See conflicted files

# Edit files to resolve conflicts
# Look for <<<<<<< HEAD markers

# After resolving
git add <resolved-files>
git commit -m "Merge develop into feature/..."
```

---

## Role-Specific Workflows

### Gabriele (Model Architect)

**Typical branches:**
- `feature/gabriele-fpn-architecture`
- `feature/gabriele-loss-function`
- `feature/gabriele-nms-implementation`

**Files to modify:**
- `src/models/phobia_net_fpn.py`
- `src/models/loss_fpn.py`
- `src/models/nms.py`

### Member A (Data Specialist)

**Typical branches:**
- `feature/memberA-dataset-merge`
- `feature/memberA-data-augmentation`
- `feature/memberA-statistics-analysis`

**Files to modify:**
- `src/data/phobia_dataset.py`
- `scripts/merge_final_dataset.py`
- `docs/DATASET_FINAL_README.md`

### Member C (Demo Engineer)

**Typical branches:**
- `feature/memberC-inference-pipeline`
- `feature/memberC-video-processing`
- `feature/memberC-streamlit-interface`

**Files to modify:**
- `src/inference/predictor.py`
- `src/demo/video_processor.py`
- `demo_app.py`

---

## Common Scenarios

### Scenario 1: Quick Fix on Main

```bash
# Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/fix-critical-bug

# Make fix
# ... edit ...

# Commit and push
git commit -am "fix: Correct class count in config"
git push origin hotfix/fix-critical-bug

# Merge to main AND develop
git checkout main
git merge hotfix/fix-critical-bug
git push origin main

git checkout develop
git merge hotfix/fix-critical-bug
git push origin develop

# Delete hotfix branch
git branch -d hotfix/fix-critical-bug
```

### Scenario 2: Sharing Work in Progress

```bash
# In your feature branch
git add .
git commit -m "WIP: Implementing P3 scale detection"
git push origin feature/gabriele-fpn-p3

# Share with team
# Teammate can check out:
git fetch origin
git checkout -b gabriele-fpn-p3 origin/feature/gabriele-fpn-p3
```

### Scenario 3: Updating Feature from Develop

```bash
# Someone merged important changes to develop
# Update your feature branch

git checkout feature/your-branch
git fetch origin
git merge origin/develop

# Or use rebase for cleaner history (advanced)
git rebase origin/develop
```

---

## Emergency Procedures

### Undo Last Commit (Not Pushed)

```bash
# Keep changes, undo commit
git reset --soft HEAD~1

# Discard changes completely
git reset --hard HEAD~1
```

### Undo Pushed Commit

```bash
# Create revert commit (safe)
git revert HEAD
git push origin develop
```

### Recover Deleted Branch

```bash
# Find commit hash
git reflog

# Recreate branch
git checkout -b feature/recovered-branch <commit-hash>
```

### Discard All Local Changes

```bash
# Nuclear option - use with caution!
git reset --hard HEAD
git clean -fd
```

---

## Best Practices

### DO ✅

1. **Pull before push**: Always sync before pushing
2. **Small commits**: One logical change per commit
3. **Test before merge**: Run tests locally
4. **Write meaningful messages**: Explain WHY, not just WHAT
5. **Delete merged branches**: Keep repo clean

### DON'T ❌

1. **Don't commit directly to main**: Always use feature branches
2. **Don't commit secrets**: No API keys, passwords
3. **Don't commit large files**: Use Git LFS or Drive for datasets
4. **Don't force push to shared branches**: `git push -f` breaks others' work
5. **Don't commit commented code**: Delete instead of commenting

---

## .gitignore

Our `.gitignore` excludes:

```gitignore
# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# Data
data/
*.zip
*.tar.gz

# Models
outputs/checkpoints/
*.pth
*.pt

# IDE
.vscode/
.idea/

# Experiments
runs/
logs/
```

---

## Cheat Sheet

```bash
# Common commands
git status                          # Check status
git branch                          # List branches
git checkout -b feature/name        # Create branch
git add .                           # Stage all changes
git commit -m "message"             # Commit
git push origin feature/name        # Push branch
git pull origin develop             # Pull develop
git merge develop                   # Merge develop
git branch -d feature/name          # Delete local branch
git push origin --delete feature/name  # Delete remote branch

# Undo/Fix
git reset --soft HEAD~1             # Undo last commit (keep changes)
git revert HEAD                     # Revert last commit (safe)
git stash                           # Save changes temporarily
git stash pop                       # Restore stashed changes

# Info
git log --oneline                   # View history
git diff                            # View changes
git remote -v                       # View remotes
```

---

## Resources

- Official Git docs: https://git-scm.com/doc
- Git branching model: https://nvie.com/posts/a-successful-git-branching-model/
- Conventional commits: https://www.conventionalcommits.org/

---

## Team Contacts

See `docs/TEAM_ROLES.md` for team member responsibilities and contact info.
