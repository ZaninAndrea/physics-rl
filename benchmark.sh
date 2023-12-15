echo "------------------------------------"
echo "FIRST RUN"
echo "------------------------------------"
echo "Running benchmark with N=1"
echo "N=1" >> times.txt
echo "N=1" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 1 python example_ms.py >> logs.txt

echo "Running benchmark with N=2"
echo "N=2" >> times.txt
echo "N=2" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 2 python example_ms.py >> logs.txt

echo "Running benchmark with N=4"
echo "N=4" >> times.txt
echo "N=4" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 4 python example_ms.py >> logs.txt

echo "Running benchmark with N=8"
echo "N=8" >> times.txt
echo "N=8" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 8 python example_ms.py >> logs.txt

echo "------------------------------------"
echo "SECOND RUN"
echo "------------------------------------"
echo "Running benchmark with N=1"
echo "N=1" >> times.txt
echo "N=1" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 1 python example_ms.py >> logs.txt

echo "Running benchmark with N=2"
echo "N=2" >> times.txt
echo "N=2" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 2 python example_ms.py >> logs.txt

echo "Running benchmark with N=4"
echo "N=4" >> times.txt
echo "N=4" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 4 python example_ms.py >> logs.txt

echo "Running benchmark with N=8"
echo "N=8" >> times.txt
echo "N=8" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 8 python example_ms.py >> logs.txt

echo "------------------------------------"
echo "THIRD RUN"
echo "------------------------------------"
echo "Running benchmark with N=1"
echo "N=1" >> times.txt
echo "N=1" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 1 python example_ms.py >> logs.txt

echo "Running benchmark with N=2"
echo "N=2" >> times.txt
echo "N=2" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 2 python example_ms.py >> logs.txt

echo "Running benchmark with N=4"
echo "N=4" >> times.txt
echo "N=4" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 4 python example_ms.py >> logs.txt

echo "Running benchmark with N=8"
echo "N=8" >> times.txt
echo "N=8" >> logs.txt
/usr/bin/time -ao times.txt mpirun -n 8 python example_ms.py >> logs.txt