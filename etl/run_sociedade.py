import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASSOS = ["extract", "transform", "validate", "load"]


def parse_args():
    ate, desde = None, None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--ate"   and i + 1 < len(args): ate   = args[i + 1]
        if arg == "--desde" and i + 1 < len(args): desde = args[i + 1]
    return desde, ate


def correr_passo(nome: str) -> bool:
    t0 = time.time()
    print(f"\n{'='*55}\n  PASSO: {nome.upper()}\n{'='*55}")

    if nome == "extract":
        from etl.extract_sociedade import main
    elif nome == "transform":
        from etl.transform_sociedade import main
    elif nome == "validate":
        from etl.validate_sociedade import main
    elif nome == "load":
        from etl.load_sociedade import main
    else:
        print(f"  Passo desconhecido: {nome}")
        return False

    try:
        main()
        print(f"\n  → {nome} concluído em {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ✗ Erro em {nome}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    desde, ate = parse_args()
    idx_desde  = PASSOS.index(desde) if desde in PASSOS else 0
    idx_ate    = PASSOS.index(ate)   if ate   in PASSOS else len(PASSOS) - 1
    passos     = PASSOS[idx_desde : idx_ate + 1]

    print(f"\n{'='*55}")
    print(f"  CIM Lezíria do Tejo · Cluster 6 — Sociedade")
    print(f"  Pipeline: {' → '.join(p.upper() for p in passos)}")
    print(f"{'='*55}")

    t_total = time.time()
    for passo in passos:
        if not correr_passo(passo):
            print(f"\n  Pipeline interrompido em '{passo}'.")
            sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  ✓ Pipeline completo em {time.time() - t_total:.1f}s")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
