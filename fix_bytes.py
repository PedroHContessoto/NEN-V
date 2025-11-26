#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrige double encoding substituindo sequências de bytes incorretas
"""

import sys
from pathlib import Path

# Mapeamento de bytes incorretos (double UTF-8) para bytes corretos (UTF-8 simples)
BYTE_FIXES = {
    # á: de \xc3\x83\xc2\xa1 para \xc3\xa1
    b'\xc3\x83\xc2\xa1': b'\xc3\xa1',
    # ã: de \xc3\x83\xc2\xa3 para \xc3\xa3
    b'\xc3\x83\xc2\xa3': b'\xc3\xa3',
    # ç: de \xc3\x83\xc2\xa7 para \xc3\xa7
    b'\xc3\x83\xc2\xa7': b'\xc3\xa7',
    # é: de \xc3\x83\xc2\xa9 para \xc3\xa9
    b'\xc3\x83\xc2\xa9': b'\xc3\xa9',
    # í: de \xc3\x83\xc2\xad para \xc3\xad
    b'\xc3\x83\xc2\xad': b'\xc3\xad',
    # ó: de \xc3\x83\xc2\xb3 para \xc3\xb3
    b'\xc3\x83\xc2\xb3': b'\xc3\xb3',
    # ô: de \xc3\x83\xc2\xb4 para \xc3\xb4
    b'\xc3\x83\xc2\xb4': b'\xc3\xb4',
    # õ: de \xc3\x83\xc2\xb5 para \xc3\xb5
    b'\xc3\x83\xc2\xb5': b'\xc3\xb5',
    # ú: de \xc3\x83\xc2\xba para \xc3\xba
    b'\xc3\x83\xc2\xba': b'\xc3\xba',
    # Â (standalone): de \xc3\x82 para nada (remover)
    b'\xc3\x82': b'',
}

def fix_bytes_in_file(file_path):
    """
    Substitui sequências de bytes incorretas por corretas
    """
    try:
        # Lê bytes raw
        with open(file_path, 'rb') as f:
            content_bytes = f.read()

        original_bytes = content_bytes
        replacements_made = 0

        # Aplica todas as substituições de bytes
        for wrong_bytes, correct_bytes in BYTE_FIXES.items():
            count_before = content_bytes.count(wrong_bytes)
            if count_before > 0:
                content_bytes = content_bytes.replace(wrong_bytes, correct_bytes)
                replacements_made += count_before

        # Se mudou algo, salva
        if content_bytes != original_bytes:
            with open(file_path, 'wb') as f:
                f.write(content_bytes)

            return True, f"Fixed ({replacements_made} byte sequences)"
        else:
            return False, "No double encoding found"

    except Exception as e:
        return False, f"Error: {str(e)[:30]}"

def main():
    base_path = Path(__file__).parent
    dirs_to_process = [
        base_path / 'src',
        base_path / 'examples',
    ]

    print("=" * 70)
    print("Corrigindo double encoding UTF-8 (byte-level)")
    print("=" * 70)

    total_files = 0
    fixed_files = 0
    total_replacements = 0

    for directory in dirs_to_process:
        if not directory.exists():
            continue

        print(f"\n[*] Processando: {directory}")
        print("-" * 70)

        for file_path in sorted(directory.rglob('*.rs')):
            total_files += 1
            success, message = fix_bytes_in_file(file_path)

            status = "[OK]" if success else "[--]"
            relative_path = file_path.relative_to(base_path)
            print(f"{status} {str(relative_path):45s} {message}")

            if success:
                fixed_files += 1
                # Extrai número de substituições
                if "byte sequences" in message:
                    num = int(message.split('(')[1].split()[0])
                    total_replacements += num

    print("\n" + "=" * 70)
    print(f"Resumo:")
    print(f"  Total de arquivos: {total_files}")
    print(f"  Arquivos corrigidos: {fixed_files}")
    print(f"  Total de substituicoes: {total_replacements}")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
