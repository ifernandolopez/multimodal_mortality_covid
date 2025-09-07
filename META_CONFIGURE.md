This file provides supplementary instructions for installing and configuring environment managers  (pyenv, conda, uv).

For the main project setup, see `CONFIGURE.md`.

Use the steps here only if these tools are not already available on your system.

## Install and configure pyenv

Clone the repository:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

Add it to your shell configuration (~/.profile):

```bash
# ~/.profile
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Reload your shell or run source ~/.profile for the changes to take effect.

## Configure conda

If conda is already installed system-wide, you may need to update your environment variables so it is accessible:

```bash
export PATH="/opt/tljh/user/condabin:$PATH"
source /opt/tljh/user/etc/profile.d/conda.sh
```

## Install and configure uv

Download the latest binary:

```bash
mkdir -p ~/.local/bin
cd ~/.local/bin
wget https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz
rm uv-x86_64-unknown-linux-gnu.tar.gz
cd ~/.local/bin/uv-x86_64-unknown-linux-gnu
chmod +x uv
```

Add `uv` to the PATH:

```bash
export PATH="$HOME/.local/bin/uv-x86_64-unknown-linux-gnu:$PATH"
```

Now uv will be available as a command in your terminal.





