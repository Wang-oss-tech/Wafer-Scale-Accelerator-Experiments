set -e
P=$1
L=$2
fabric_w=$(($1 + $2 + 7)) # P + L
fabric_h=$(($1 + 2)) # P

Mt=$(($3 / $1))
Nt=$(($4 / $1))

echo "P=$1, L=$2, M=$3, N=$4"

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$fabric_w","$fabric_h" --fabric-offsets=4,1 \
    --params=P:"$P",L:"$L",Mt:"$Mt",Nt:"$Nt" \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P "$P" --L "$L" --M "$3" --N "$4"