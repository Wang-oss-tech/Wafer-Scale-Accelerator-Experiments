set -e
P_1=$1
P_2=$2 # P_2 > P_1
fabric_w=$(($2 + 7))
fabric_h=$(($2 + 2))

Mt_1=$(($3 / $1))
Nt_1=$(($4 / $1))

Mt_2=$(($3 / $2))
Nt_2=$(($4 / $2))

echo "P_1=$1, P_2=$2, M=$3, N=$4"

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$fabric_w","$fabric_h" --fabric-offsets=4,1 \
    --params=P_1:"$P_1",P_2:"$P_2",Mt_1:"$Mt_1",Nt_1:"$Nt_1",Mt_2:"$Mt_2",Nt_2:"$Nt_2" \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P_1 "$P_1" --P_2 "$P_2" --M "$3" --N "$4"