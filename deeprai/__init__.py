from colorama import Fore, Style
import os
for file in os.listdir('.'):
    if file.endswith('.so') or file.endswith('.pyd'):
        src = os.path.join(os.getcwd(), file)
        dst = os.path.join(os.getcwd(), 'deeprai/engine/cython', file)
        os.rename(src, dst)
print(Fore.CYAN + Style.BRIGHT  + "Loaded DeeprAI 0.0.7 BETA")
Style.RESET_ALL
Fore.WHITE