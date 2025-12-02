# 🔀 Git Workflow Guide - PhobiaShield

Guida completa per lavorare in team con Git e GitHub.

## 📋 Indice

1. [Setup Iniziale](#setup-iniziale)
2. [Workflow Quotidiano](#workflow-quotidiano)
3. [Branch Strategy](#branch-strategy)
4. [Commit Messages](#commit-messages)
5. [Pull Requests](#pull-requests)
6. [Risoluzione Conflitti](#risoluzione-conflitti)
7. [Best Practices](#best-practices)

---

## 1️⃣ Setup Iniziale

### Primo Setup (solo una volta)

```bash
# 1. Clone della repository
git clone https://github.com/your-team/PhobiaShield.git
cd PhobiaShield

# 2. Configura il tuo nome e email
git config --global user.name "Il Tuo Nome"
git config --global user.email "tua-email@example.com"

# 3. Crea il tuo ambiente virtuale
conda create -n phobiashield python=3.10
conda activate phobiashield

# 4. Installa dipendenze
pip install -r requirements.txt
pip install -e .

# 5. Configura W&B
wandb login
```

---

## 2️⃣ Workflow Quotidiano

### Iniziare a Lavorare

```bash
# 1. Assicurati di essere su main
git checkout main

# 2. Aggiorna main con le ultime modifiche
git pull origin main

# 3. Crea il tuo branch per la feature
# Format: feature/nome-descrittivo
git checkout -b feature/data-augmentation

# Oppure per Membro A:
git checkout -b feature/data-pipeline

# Oppure per Membro B:
git checkout -b feature/model-architecture

# Oppure per Membro C:
git checkout -b feature/inference-demo
```

### Durante il Lavoro

```bash
# Controlla lo stato dei file modificati
git status

# Aggiungi file specifici
git add src/data/dataset.py
git add src/data/augmentation.py

# Oppure aggiungi tutto (usa con cautela!)
git add .

# Fai il commit con messaggio descrittivo
git commit -m "feat: implement PhobiaDataset class"

# Push del tuo branch su GitHub
git push origin feature/data-pipeline

# Se è il primo push del branch
git push -u origin feature/data-pipeline
```

### Aggiornare il Tuo Branch con Main

```bash
# Mentre sei sul tuo branch
git checkout feature/data-pipeline

# Fetch delle ultime modifiche
git fetch origin

# Merge di main nel tuo branch
git merge origin/main

# Oppure usa rebase (più pulito, ma più avanzato)
git rebase origin/main

# Se ci sono conflitti, risolvili e poi:
git add .
git commit -m "merge: resolve conflicts with main"
git push origin feature/data-pipeline
```

### Completare il Lavoro

```bash
# 1. Assicurati che tutto sia committato
git status

# 2. Push finale
git push origin feature/data-pipeline

# 3. Apri Pull Request su GitHub
# Vai su: https://github.com/your-team/PhobiaShield/pulls
# Click "New Pull Request"
# Seleziona: base=main, compare=feature/data-pipeline
# Aggiungi descrizione e reviewers
```

---

## 3️⃣ Branch Strategy

### Branch Principale

- **`main`**: Branch principale, sempre stabile
  - Contiene solo codice testato e funzionante
  - Nessuno fa commit direttamente su main
  - Aggiornato solo tramite Pull Request

### Branch per Membro

Ogni membro ha il suo branch principale per la sua area:

- **`feature/data-pipeline`**: Membro A (Data Specialist)
- **`feature/model-architecture`**: Membro B (Model Architect)
- **`feature/inference-demo`**: Membro C (Deployment Engineer)

### Branch per Sub-Task

Opzionalmente, creare branch più specifici:

```bash
# Da feature/data-pipeline
git checkout feature/data-pipeline
git checkout -b feature/data-augmentation-advanced

# Da feature/model-architecture
git checkout feature/model-architecture
git checkout -b feature/loss-function-debug

# Da feature/inference-demo
git checkout feature/inference-demo
git checkout -b feature/streamlit-ui
```

---

## 4️⃣ Commit Messages

### Convention: Conventional Commits

Format: `<type>: <description>`

#### Types

- **feat**: Nuova feature
  ```bash
  git commit -m "feat: add PhobiaDataset class"
  git commit -m "feat: implement custom loss function"
  git commit -m "feat: add NMS algorithm"
  ```

- **fix**: Bug fix
  ```bash
  git commit -m "fix: correct bbox coordinate conversion"
  git commit -m "fix: resolve dimension mismatch in forward pass"
  ```

- **docs**: Documentazione
  ```bash
  git commit -m "docs: add docstrings to dataset class"
  git commit -m "docs: update README with installation steps"
  ```

- **refactor**: Refactoring codice
  ```bash
  git commit -m "refactor: simplify data loading logic"
  git commit -m "refactor: extract NMS to separate function"
  ```

- **test**: Aggiunta test
  ```bash
  git commit -m "test: add unit tests for loss function"
  ```

- **chore**: Maintenance
  ```bash
  git commit -m "chore: update requirements.txt"
  git commit -m "chore: add .gitignore rules"
  ```

#### Multi-line Commits (per cambiamenti complessi)

```bash
git commit -m "feat: implement data augmentation pipeline

- Add horizontal flip with 50% probability
- Add brightness/contrast adjustment
- Add Gaussian noise and blur
- Integrate with PhobiaDataset
- Add config file for augmentation parameters"
```

---

## 5️⃣ Pull Requests

### Creare una Pull Request

1. **Push del tuo branch**
   ```bash
   git push origin feature/data-pipeline
   ```

2. **Su GitHub**
   - Vai a: https://github.com/your-team/PhobiaShield
   - Click "Pull requests" → "New pull request"
   - Base: `main`, Compare: `feature/data-pipeline`
   - Click "Create pull request"

3. **Compila la PR**
   ```markdown
   ## Description
   Implements PhobiaDataset class with YOLO format support
   
   ## Changes
   - ✅ Created PhobiaDataset class in src/data/dataset.py
   - ✅ Added YOLO annotation parsing
   - ✅ Integrated data augmentation
   - ✅ Added unit tests
   
   ## Testing
   - [x] Tested with dummy data
   - [x] Verified with actual COCO subset
   - [ ] Performance benchmarking (TODO)
   
   ## Related Issues
   Closes #1
   ```

4. **Assegna Reviewers**
   - Assegna gli altri membri del team come reviewers
   - Aspetta l'approvazione prima del merge

### Processo di Review

#### Per il Reviewer:

1. **Leggi il codice**
   - Vai su "Files changed"
   - Leggi i cambiamenti
   - Aggiungi commenti dove necessario

2. **Testa localmente** (opzionale ma consigliato)
   ```bash
   git fetch origin
   git checkout feature/data-pipeline
   python -m pytest  # Run tests
   ```

3. **Approva o Request Changes**
   - Approva: "Looks good to me! (LGTM)"
   - Request changes: Spiega cosa deve essere modificato

#### Per l'Autore:

1. **Rispondi ai commenti**
2. **Fai le modifiche richieste**
3. **Push degli aggiornamenti**
   ```bash
   git add .
   git commit -m "fix: address review comments"
   git push origin feature/data-pipeline
   ```

### Merge della PR

**Opzione 1: Merge Commit** (consigliato per team)
```
feature/data-pipeline → main
(mantiene tutta la history)
```

**Opzione 2: Squash and Merge**
```
Combina tutti i commit in uno solo
(history più pulita)
```

**Opzione 3: Rebase and Merge**
```
Applica i commit uno per uno su main
(history lineare)
```

Per il nostro progetto, usa **Merge Commit**.

---

## 6️⃣ Risoluzione Conflitti

### Scenario: Conflict durante Merge

```bash
# Situazione: stai mergendo main nel tuo branch
git merge origin/main

# Output:
# Auto-merging src/data/dataset.py
# CONFLICT (content): Merge conflict in src/data/dataset.py
# Automatic merge failed; fix conflicts and then commit the result.
```

### Risoluzione

1. **Apri i file in conflitto**
   ```python
   # src/data/dataset.py
   
   <<<<<<< HEAD (current change - il tuo codice)
   def load_data(self):
       return self.data
   =======
   def load_data(self, normalize=True):
       return self.normalize(self.data) if normalize else self.data
   >>>>>>> origin/main (incoming change - da main)
   ```

2. **Scegli cosa tenere**
   ```python
   # Opzione A: Tieni il tuo codice
   def load_data(self):
       return self.data
   
   # Opzione B: Tieni il codice da main
   def load_data(self, normalize=True):
       return self.normalize(self.data) if normalize else self.data
   
   # Opzione C: Combina entrambi (spesso la migliore)
   def load_data(self, normalize=True):
       # Tua logica + logica da main
       data = self.data
       return self.normalize(data) if normalize else data
   ```

3. **Rimuovi i markers di conflitto**
   ```python
   # Rimuovi: <<<<<<< HEAD, =======, >>>>>>> origin/main
   ```

4. **Stage, commit, push**
   ```bash
   git add src/data/dataset.py
   git commit -m "merge: resolve conflicts in dataset.py"
   git push origin feature/data-pipeline
   ```

### Prevenire Conflitti

1. **Pull frequentemente**
   ```bash
   # Ogni mattina prima di iniziare
   git checkout main
   git pull origin main
   git checkout feature/data-pipeline
   git merge main
   ```

2. **Comunica con il team**
   - "Sto lavorando su dataset.py"
   - "Non modificate loss.py, ci sto lavorando io"

3. **Lavora su file diversi**
   - Membro A: `src/data/`
   - Membro B: `src/models/`
   - Membro C: `src/inference/`

---

## 7️⃣ Best Practices

### DO's ✅

- ✅ **Commit piccoli e frequenti**
  ```bash
  git commit -m "feat: add bbox validation"
  git commit -m "feat: add bbox clipping"
  # Meglio di un singolo commit enorme
  ```

- ✅ **Pull prima di push**
  ```bash
  git pull origin main  # Prima
  git push origin feature/data-pipeline  # Poi
  ```

- ✅ **Branch descrittivi**
  ```bash
  feature/loss-function-implementation  # ✅ Good
  fix-stuff  # ❌ Bad
  ```

- ✅ **Testa prima di committare**
  ```bash
  python -m pytest
  python src/data/dataset.py  # Test manuale
  git commit -m "feat: add dataset class"
  ```

- ✅ **Clear nei notebook prima di committare**
  ```bash
  # In Jupyter: Kernel → Restart & Clear Output
  git add notebooks/training_colab.ipynb
  git commit -m "docs: add training notebook"
  ```

### DON'Ts ❌

- ❌ **NON committare file pesanti**
  ```bash
  # NO:
  git add data/images/*.jpg  # ❌
  git add outputs/checkpoints/*.pth  # ❌
  git add *.mp4  # ❌
  
  # Usa .gitignore invece!
  ```

- ❌ **NON committare credenziali**
  ```bash
  # NO:
  wandb_api_key = "abc123"  # ❌
  
  # USA:
  import os
  wandb_api_key = os.getenv("WANDB_API_KEY")  # ✅
  ```

- ❌ **NON fare commit direttamente su main**
  ```bash
  git checkout main
  git add .
  git commit -m "stuff"  # ❌ MAI!
  ```

- ❌ **NON pushare codice che non compila**
  ```bash
  # Test prima!
  python src/models/phobia_net.py  # Deve funzionare
  git push  # Solo se funziona
  ```

---

## 🆘 Comandi di Emergenza

### Annullare l'ultimo commit (non ancora pushato)

```bash
# Annulla commit ma mantieni i cambiamenti
git reset --soft HEAD~1

# Annulla commit E cambiamenti (ATTENZIONE!)
git reset --hard HEAD~1
```

### Annullare modifiche non committate

```bash
# Singolo file
git checkout -- src/data/dataset.py

# Tutti i file
git reset --hard HEAD
```

### Tornare a un commit precedente

```bash
# Vedi la history
git log --oneline

# Torna a un commit specifico
git checkout abc123f  # Hash del commit

# Crea un branch da quel punto
git checkout -b fix-from-old-commit
```

### Salvare lavoro temporaneo (stash)

```bash
# Salva modifiche non committate
git stash

# Cambia branch o fai altro lavoro
git checkout other-branch

# Recupera le modifiche salvate
git checkout your-branch
git stash pop
```

---

## 📞 Cheat Sheet Rapido

```bash
# Setup
git clone <url>
git config --global user.name "Nome"
git config --global user.email "email"

# Workflow base
git checkout main
git pull origin main
git checkout -b feature/my-feature
# ... lavora ...
git add .
git commit -m "feat: description"
git push origin feature/my-feature

# Update branch
git fetch origin
git merge origin/main

# Risoluzione conflitti
# 1. Apri file, risolvi conflitti
# 2. git add <file>
# 3. git commit -m "merge: resolve conflicts"
# 4. git push

# Utility
git status
git log --oneline
git branch -a
git diff
```

---

## 🎯 Workflow per Membri Specifici

### Membro A (Data Specialist)

```bash
# Setup
git checkout -b feature/data-pipeline

# Lavoro tipico
# - Modifica: src/data/dataset.py
# - Modifica: src/data/augmentation.py
# - Modifica: cfg/data/*.yaml

git add src/data/
git add cfg/data/
git commit -m "feat: implement data pipeline"
git push origin feature/data-pipeline
```

### Membro B (Model Architect)

```bash
# Setup
git checkout -b feature/model-architecture

# Lavoro tipico
# - Modifica: src/models/phobia_net.py
# - Modifica: src/models/loss.py
# - Modifica: cfg/model/*.yaml

git add src/models/
git add cfg/model/
git commit -m "feat: implement loss function"
git push origin feature/model-architecture
```

### Membro C (Deployment Engineer)

```bash
# Setup
git checkout -b feature/inference-demo

# Lavoro tipico
# - Modifica: src/inference/nms.py
# - Modifica: src/inference/video_processor.py
# - Modifica: app/streamlit_app.py

git add src/inference/
git add app/
git commit -m "feat: implement video processing"
git push origin feature/inference-demo
```

---

**Buon lavoro! 🚀**

Per domande, chiedete agli altri membri del team o consultate la [documentazione Git ufficiale](https://git-scm.com/doc).
