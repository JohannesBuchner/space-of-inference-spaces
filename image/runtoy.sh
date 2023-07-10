LOWD="asymgauss-4d corrfunnel4-2d rosenbrock-2d eggbox-2d beta-2d loggamma-2d box-5d"
MIDD="asymgauss-16d corrfunnel4-10d rosenbrock-20d beta-10d loggamma-10d"
HIGHD="asymgauss-100d corrfunnel4-50d rosenbrock-50d beta-30d loggamma-30d"
GENSIG="multisine-0comp-2d multisine-1comp-5d multisine-2comp-8d multisine-3comp-11d"
SPIKESLAB=$(echo spikeslab{1,40,1000}-2d-{4,40,400,4000} spikeslab{1,40,1000}-2d-40-off{1,2,4,10})
for p in $LOWD $GENSIG $MIDD $HIGHD $SPIKESLAB
do

#for s in vbis vbis-wide goodman-weare slice ultranest-fast multinest ultranest-safe 
#for s in goodman-weare slice ultranest-fast multinest ultranest-safe # multinest ultranest-safe 
#for s in nestle dynesty-multiell multinest
for s in $*
#for s in goodman-weare slice multinest
do
[ -e systematiclogs/$p/$s/chains ] && continue

echo
echo
echo "===== PROBLEM:$p === SAMPLER:$s ===== "
echo
echo

mkdir -p systematiclogs/$p/$s
SAMPLER=$s PROBLEM=$p python3 problems.py 2>&1 | tee systematiclogs/$p/$s/log.txt

done
#wait
#break
done
