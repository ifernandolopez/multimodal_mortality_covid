## To install pyenv:

Clone the repository:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

Configure it in .profile:

```bash
# ~/.profile
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

## To configure Conda:

Clone the repository:

```bash
export PATH="/opt/tljh/user/condabin:$PATH"
source /opt/tljh/user/etc/profile.d/conda.sh
```

## To install and configure uv:

Download the binary:

```bash
mkdir -p ~/.local/bin
cd ~/.local/bin
wget https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz
rm uv-x86_64-unknown-linux-gnu.tar.gz
cd ~/.local/bin/uv-x86_64-unknown-linux-gnu
chmod +x uv
```

Add it to the PATH:

```bash
export PATH="$HOME/.local/bin/uv-x86_64-unknown-linux-gnu:$PATH"







