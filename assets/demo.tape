# VHS tape for the README demo. Regenerate with: vhs assets/demo.tape
# Uses the offline echo backend, so it needs no API key and is deterministic.

Output assets/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 720
Set Padding 20
Set Theme "Catppuccin Mocha"

Hide
Type 'cd "$(git rev-parse --show-toplevel)" && source .venv/bin/activate && export PS1="$ " _ZO_DOCTOR=0 && clear'
Enter
Sleep 1.5s
Type "a2acode serve --backend echo > /tmp/demo.log 2>&1 &"
Enter
Sleep 2.5s
Type "clear"
Enter
Show

Type "a2acode call 'refactor the parser'"
Enter
Sleep 3.5s

Type "# a tool that needs approval pauses the task:"
Enter
Type "a2acode call 'sudo reboot'"
Enter
Sleep 4s
Sleep 1.5s
