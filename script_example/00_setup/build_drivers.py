import os
import subprocess

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'lira_code')
src = os.path.join(CODE, 'tools', 'mpiprof.c')
lib = os.path.join(CODE, 'tools', 'libmpiprof.so')
subprocess.run([os.environ.get('LIRA_MPICC', 'cc'), '-O2', '-fPIC', '-shared', '-o', lib, src], check=True)
dyn = subprocess.run(['readelf', '-d', lib], stdout=subprocess.PIPE, universal_newlines=True).stdout
rpath = [line.strip() for line in dyn.splitlines() if 'RPATH' in line or 'RUNPATH' in line]
print(lib)
print('\n'.join(rpath) if rpath else 'no RPATH or RUNPATH')
