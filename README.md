# YoungCC

YoungCC는 NumPy와 SciPy로 구현한 독립적인 닫힌껍질 전자구조 계산 프로그램입니다. RHF 및 RKS 단일점 계산을 지원합니다. RKS의 `LDA`는 Slater 교환과 VWN5 상관을 사용합니다. 1차 검증 대상은 He, H₂, H₂O이며 제공 기저는 H, He, O의 STO-3G입니다.

## 설치와 실행

Python 3.11 이상이 필요합니다. Linux에서는 가상 환경을 활성화합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
youngcc examples/h2o_rhf.json -o build/h2o_rhf
youngcc examples/h2o_lda.json -o build/h2o_lda
OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

Windows에서는 [Releases](https://github.com/Skylark51/YoungCC/releases)에서 x64 실행 파일을 받을 수 있습니다.

- **YoungCC_for_Windows_GUI.exe**를 실행하면 JSON 입력을 열거나 직접 편집할 수 있습니다. **Run calculation**을 누르고 결과 JSON의 위치를 선택하면 SCF 진행이 창에 표시되고 같은 위치에 로그가 저장됩니다.
- **YoungCC_for_Windows_CLI.exe**는 PowerShell에서 실행합니다.

```powershell
.\YoungCC_for_Windows_CLI.exe examples\h2o_lda.json -o build\h2o_lda
```

소스에서 Windows 실행 파일을 빌드하려면 Python과 Tk를 설치하고 다음 명령을 실행합니다. 결과는 `dist\`에 생성됩니다.

```powershell
python -m pip install -e ".[test]" pyinstaller
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/build-windows.ps1
```

## 입력과 출력

```json
{
  "method": "LDA",
  "basis": "STO-3G",
  "units": "angstrom",
  "charge": 0,
  "multiplicity": 1,
  "atoms": [["H", 0, 0, -0.37], ["H", 0, 0, 0.37]]
}
```

좌표 단위는 `angstrom` 또는 `bohr`입니다. 전자 수는 양의 짝수여야 하며, 다중도는 1이어야 합니다. 출력 접두사를 생략하면 입력 파일명에 `.youngcc`를 붙입니다. `.log`에는 SCF 반복과 에너지 성분을, `.json`에는 입력, 에너지, 궤도 에너지, AO 밀도 행렬, 전자 수 및 수렴 이력을 저장합니다. 에너지는 Hartree입니다. 전자 수는 `Tr(PS)`로 계산하고 LDA는 수치 격자 적분값도 저장합니다. CLI 종료 코드는 수렴 시 0, 반복 한도 도달 시 1, 입력 또는 계산 오류 시 2입니다.

## Gaussian 16 비교

라이선스가 있는 Gaussian 16 환경에서 실행합니다. 다음 예시는 Bash와 환경 설정이 끝난 `g16`, `formchk`를 사용합니다.

```sh
python benchmarks/gaussian16/compare.py prepare build/gaussian16
(
  cd build/gaussian16
  for input in *.gjf; do
    stem=${input%.gjf}
    g16 < "$input" > "$stem.log" || exit 1
    formchk "$stem.chk" "$stem.fchk" || exit 1
  done
)
OPENBLAS_NUM_THREADS=1 python benchmarks/gaussian16/compare.py compare build/gaussian16
```

입력은 같은 bohr 좌표, 전하, 다중도, 기저, core guess 및 방법을 사용합니다. Gaussian은 RHF 또는 `SVWN5`와 SuperFineGrid를 사용합니다. 수렴 기준은 두 프로그램에서 서로 다른 척도를 사용하므로 수치상 동일하게 해석할 수 없습니다. `comparison.json`은 에너지, 궤도 에너지, 지정 좌표의 전자 밀도, 전자 수, 반복 횟수, 경과 시간 및 오차를 기록합니다. 밀도 비교는 동일한 STO-3G AO에서 Gaussian의 fchk 밀도 행렬을 평가합니다. 총 에너지 절대 오차가 어느 케이스에서든 0.001 Eh를 초과하면 종료 코드 1을 반환합니다. 참조 결과와 한계는 [검증 기록](benchmarks/README.md)에 있습니다.

## 1차 범위

적분 엔진은 실수 Cartesian s/p 함수와 조밀한 4차원 ERI 배열을 사용합니다. LDA는 원자 중심 Becke 분할과 자체 구적법을 사용합니다. 열린껍질, 추가 기저와 원소, 구조 최적화, 진동수, TS 탐색 및 병렬화는 후속 단계입니다. 구현 구조와 이론 출처는 [개발 노트](docs/design.md)를 참고하세요.

## 2차 기능: UB3LYP/def2-SVP/def2-TZVPP/C-PCM/D3(BJ)

추가 백엔드는 PySCF 기반입니다. 기존 RHF/LDA 엔진은 유지되며
UB3LYP/B3LYP 계산은 PySCF의 UKS 전자구조 계산을 사용합니다.

```sh
python -m pip install -e ".[advanced,test]"
youngcc examples/oh_ub3lyp.gjf -o build/oh_ub3lyp
python -m pytest -q
```

JSON과 Gaussian 형식 .gjf/.com 단일점 입력의 일부를 지원합니다.
지원하지 않는 Opt, Freq, IRC, Geom=Check, Guess=Read 등은 명시적으로 거부합니다.
출력 로그 하단에 YOUNGCC FINAL SUMMARY가 추가됩니다.
이 기능은 Gaussian 16과의 결과 일치를 아직 검증하지 않았습니다.

자세한 범위와 Gaussian 공식 문서 참조는 [advanced-backend.md](docs/advanced-backend.md)를 확인하세요.
기존 문서의 1차 범위 설명은 자체 구현 RHF/LDA 엔진을 대상으로 합니다.


## 병렬 에이전트 개발을 위한 모듈 구조

화학 계산 기초 모듈(chemistry), 입력(inputs), 자체 엔진(engines/native),
PySCF 백엔드(engines/pyscf_backend), 결과 출력(reporting), UI(interfaces)를
각각 별도 패키지로 분리했다. 기존 opendft.* 임포트와 CLI 명령은
호환 계층으로 유지한다. 실제 기준 입력/출력은 복제하지 않는다.

각 폴더의 변경 책임, 독립 브랜치/워크트리 관리, 에이전트별 작업 지시:
[Multi-agent workspaces](docs/multi-agent-workspaces.md). 루트 [AGENTS.md](AGENTS.md) 참고.
