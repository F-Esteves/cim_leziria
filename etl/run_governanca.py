import sys
import time
from pathlib import Path

# Garantir que o root do projecto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

PASSOS = ["extract", "transform", "validate", "load"]


def parse_args():
    ate    = None
    desde  = None
    args   = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--ate"    and i + 1 < len(args): ate   = args[i + 1]
        if arg == "--desde"  and i + 1 < len(args): desde = args[i + 1]
    return desde, ate


def correr_passo(nome: str):
    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  PASSO: {nome.upper()}")
    print(f"{'='*55}")

    if nome == "extract":
        from etl.extract_governanca import main
    elif nome == "transform":
        from etl.transform_governanca import main
    elif nome == "validate":
        from etl.validate_governanca import main
    elif nome == "load":
        from etl.load_governanca import main
    else:
        print(f"  Passo desconhecido: {nome}")
        return False

    try:
        main()
        elapsed = time.time() - t0
        print(f"\n  → {nome} concluído em {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ✗ Erro em {nome}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    desde, ate = parse_args()

    # Determinar quais passos correr
    idx_desde = PASSOS.index(desde) if desde in PASSOS else 0
    idx_ate   = PASSOS.index(ate)   if ate   in PASSOS else len(PASSOS) - 1
    passos    = PASSOS[idx_desde : idx_ate + 1]

    print(f"\n{'='*55}")
    print(f"  CIM Lezíria do Tejo · Cluster 1 — Governança")
    print(f"  Pipeline: {' → '.join(p.upper() for p in passos)}")
    print(f"{'='*55}")

    t_total = time.time()
    for passo in passos:
        ok = correr_passo(passo)
        if not ok:
            print(f"\n  Pipeline interrompido em '{passo}'.")
            sys.exit(1)

    elapsed = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"  ✓ Pipeline completo em {elapsed:.1f}s")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
