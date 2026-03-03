set -e

P=$1
L=$2
fabric_w=$(($1 + $2 + 7)) # P + L
fabric_h=$(($1 + 2)) # P

Mt=$(($3 / $1))
Nt=$(($4 / $1))

simulator=false

if [ -n "$5" ]; then
    simulator=$5
fi

echo "P=$1, L=$2, M=$3, N=$4, simulator=$simulator"

python compile.py "$P" "$L" "$Mt" "$Nt" "$simulator"

if [ "$simulator" == "true" ]; then
    python launch_wse3.py --P "$1" --L "$2" --M "$3" --N "$4" --simulator
else
    python launch_wse3.py --P "$1" --L "$2" --M "$3" --N "$4"
fi