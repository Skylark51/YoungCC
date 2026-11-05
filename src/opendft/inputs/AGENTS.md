# Input grammar agent
Edit only this folder, plus dedicated tests/inputs/ if needed.
Own Molecule validation, JSON and Gaussian .gjf/.com parsing.
Use https://gaussian.com/ as primary syntax authority.
Unsupported features (e.g. Opt=TS, checkpoint restart) must fail explicitly
until implemented. Do not silently reinterpret requests as single points.
Changes to Molecule fields require coordinator review.
