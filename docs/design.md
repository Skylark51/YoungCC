# 1차 설계

구현 순서는 입력 검증, STO-3G 전개, 해석적 적분, RHF, LDA 격자와 RKS, 출력과 Windows 인터페이스, 외부 참조 비교다.

| 모듈 | 책임 |
|---|---|
| input | JSON 검증, 내부 bohr 좌표와 전자 수 |
| basis | 정규화된 Cartesian 수축 GTO와 좌표별 AO 값 |
| integrals | Hermite 재귀, Boys 함수, 1전자 적분, 8중 대칭 ERI |
| scf | 일반화 고유값 문제, 닫힌껍질 밀도, Pulay DIIS와 수렴 검사 |
| dft / grid | Slater + VWN5 및 해석적 XC 퍼텐셜, Becke 분할 구적 |
| output / cli / gui | 공통 로그와 JSON, 사용자 인터페이스 |

RHF의 Fock 행렬은 `H + J - K/2`다. RKS는 `H + J + Vxc`를 사용하며, 전자 에너지는 `Tr(PH) + Tr(PJ)/2 + Exc`다. 초기 밀도는 core Hamiltonian에서 구한다. 수렴 판정에는 에너지 변화량, 실제 Fock 행렬에서 다시 구한 밀도와의 RMS 차이, AO 교환자 잔차를 모두 사용한다. 격자는 원자당 radial Gauss–Legendre 80점, polar Gauss–Legendre 24점, azimuth 균일 48점이다.

C++/Fortran 최적화 시 `build_integrals` 경계에서 배열 계약을 유지하면 SCF와 입출력을 재사용할 수 있다. ERI의 메모리는 기저 크기의 4제곱에 비례한다.

## 출처

- H, He, O STO-3G 파라미터. [Basis Set Exchange](https://www.basissetexchange.org/)
- Hermite Gaussian 적분 전개. [McMurchie와 Davidson, 1978](https://doi.org/10.1016/0021-9991(78)90092-X)
- 원자 중심 분할. [Becke, 1988](https://doi.org/10.1063/1.454033)
- VWN 상관. [Vosko, Wilk와 Nusair, 1980](https://doi.org/10.1139/p80-159)
