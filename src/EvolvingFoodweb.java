

import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.CombinationSampler;
import org.apache.commons.rng.sampling.distribution.MarsagliaTsangWangDiscreteSampler.Binomial;
import org.apache.commons.rng.sampling.distribution.NormalizedGaussianSampler;
import org.apache.commons.rng.sampling.distribution.SharedStateDiscreteSampler;
import org.apache.commons.rng.sampling.distribution.ZigguratNormalizedGaussianSampler;
import org.apache.commons.rng.simple.RandomSource;

import java.io.*;
import java.util.Arrays;


/* class EvolvingMetacommunity
 * loops over cycles (time steps) of repType and dispersal
 * writes output to file */
public class EvolvingFoodweb {

    static Comm comm;
    static Evol evol;
    static Run run;

    static Sites sites;

    public static void main(String[] args) throws IOException {

        comm = new Comm();
        evol = new Evol();
        run = new Run();
        if (args.length > 0)
            Reader.readInput(args[0], comm, evol, run);

        try (PrintWriter streamOut = new PrintWriter(new FileWriter(run.fileName))) {

            long startTime = System.currentTimeMillis();
            logTitles(streamOut);

            for (int r = 0; r < run.runs; r++)
                    for (int es = 0; es < comm.envStep.length; es++)
                        for (int dr = 0; dr < comm.dispRate.length; dr++) {

                            System.out.format("run = %d; disp = %.4f; step = %.4f%n",
                                    (r + 1), comm.dispRate[dr], comm.envStep[es]);

                            comm.init(run);
                            evol.init(comm);
                            Auxils.init(evol);
                            Init init = new Init(comm);

                            sites = new Sites(comm, evol, init, es, dr);

                            System.out.format("  time = %d; resource = %f; N = %d; mass = %f; fit = %f%n", 0, Auxils.arraySum(sites.resource), sites.foodwebSize(), sites.foodwebMass(), sites.fitnessMean());
                            logResults(0, streamOut, r, es, dr);

                            for (int t = 0; t < run.timeSteps/run.dt; t++) {
                                if (((t+1)*run.dt > run.preChange) && ((t+1)*run.dt <= (run.timeSteps - run.postChange)))
                                    sites.changeEnvironment();
                                sites.uptake();
                                sites.updatePrey();
                                sites.contributionAdults();
                                sites.reproduction();

                                if (((t+1)*run.dt == 1) || (((t + 1)*run.dt) % run.printSteps) == 0) {
                                    System.out.format("  time = %f; resource = %f; N = %d; mass = %f; fit = %f%n", (t + 1)*run.dt, Auxils.arraySum(sites.resource), sites.foodwebSize(), sites.foodwebMass(), sites.fitnessMean());
                                }
                                if (((t+1)*run.dt == 1) || (((t + 1)*run.dt) % run.saveSteps) == 0) {
                                    logResults((t + 1)*run.dt, streamOut, r, es, dr);
                                }
                            }
                        }

            long endTime = System.currentTimeMillis();
            System.out.println("EvolMetac took " + (endTime - startTime) +
                    " milliseconds.");
        }
    }

    static void logTitles(PrintWriter out) {
        out.print("gridX;gridY;patches;e_step;m;rho;microsites;nbrLoci;sigma_z;mu;omega_e;d;"
                + "run;time;patch;X;Y;environment;resource;species;bodymass;N;biomass;"
                + "genotype_mean;genotype_var;phenotype_mean;phenotype_var;"
                + "fitness_mean;fitness_var;fitness_geom;load_mean;load_var;S_mean"
                + "genotype_meta_var;phenotype_meta_var");
        out.println("");
    }

    static void logResults(double t, PrintWriter out, int r, int es, int dr) {
        for (int p = 0; p < comm.nbrPatches; p++)
            for (int s = 0; s < comm.nbrSpecies; s++) {
                out.format("%d;%d;%d;%f;%f;%f;%d;%d;%f;%f;%f;%f",
                        comm.gridX, comm.gridY, comm.nbrPatches, comm.envStep[es], comm.dispRate[dr], comm.rho, comm.microsites, evol.allLoci, evol.sigmaZ, evol.mutationRate, evol.omegaE, comm.d);
                out.format(";%d;%f;%d;%d;%d;%f",
                        r + 1, t, p + 1, comm.patchXY[p][0], comm.patchXY[p][1], sites.environment[p]);
                out.format(";%f;%d;%f;%d;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f",
                        sites.resource[p], s + 1, comm.bodyMassClass[comm.massClassSpecies[s]], sites.popSize(p, s), sites.popMass(p, s),
                        sites.genotypeMean(p, s), sites.genotypeVar(p, s), sites.phenotypeMean(p, s), sites.phenotypeVar(p, s),
                        sites.fitnessMean(p, s), sites.fitnessVar(p, s), sites.fitnessGeom(p, s), sites.loadMean(p, s), sites.loadVar(p, s), sites.selectionDiff(p, s),
                        sites.genotypeVar(s), sites.phenotypeVar(s));
                out.println("");
            }
    }
}




/* class Sites
 * keeps track of individuals and their attributes in microsites (microhabitats within patches)
 * implements repType (with inheritance and mutation) and dispersal */
class Sites {
    Comm comm;
    Evol evol;
    int totSites;

    int esPos;
    int drPos;

    int[] patch;
    int[] species;
    boolean[] alive;
    double[] mortality;

    double[] phenotype;
    double[] fitness;
    byte[][] genotype;

    double[] environment;
    int[][] massAbundance;
    double[] resource;
    double[][] uptakePrey;
    double[][] consumption;
    int[][] nbrPrey;
    int[][][] preyPos;

    int[] nbrEmpty;
    int[][] emptyPos;
    double[] nbrNewborns;
    double[][] mothersCumProb;
    double[][][] fathersCumProb;

    public Sites(Comm cmm, Evol evl, Init init, int es, int dr) {
        comm = cmm;
        evol = evl;
        esPos = es;
        drPos = dr;

        comm.calcDispNeighbours(drPos);

        totSites = comm.nbrPatches * comm.microsites;

        patch = new int[totSites];
        species = new int[totSites];
        alive = new boolean[totSites];
        mortality = new double[totSites];

        phenotype = new double[totSites];
        fitness = new double[totSites];
        genotype = new byte[totSites][2 * evol.allLoci];

        environment = new double[comm.nbrPatches];
        resource = new double[comm.nbrPatches];
        massAbundance = new int[comm.nbrPatches][comm.trophicLevels];
        uptakePrey = new double[comm.nbrPatches][comm.trophicLevels];
        consumption = new double[comm.nbrPatches][comm.trophicLevels];
        nbrPrey = new int[comm.nbrPatches][comm.trophicLevels];
        preyPos = new int[comm.nbrPatches][comm.trophicLevels][comm.microsites];

        nbrEmpty = new int[comm.nbrPatches];
        emptyPos = new int[comm.nbrPatches][comm.microsites];
        nbrNewborns = new double[comm.nbrPatches];
        mothersCumProb = new double[comm.nbrPatches][totSites];
        fathersCumProb = new double[comm.nbrPatches][comm.nbrSpecies][comm.microsites];

        double indGtp;
        Arrays.fill(alive, false);
        Arrays.fill(mortality, 1);

        for (int p = 0; p < comm.nbrPatches; p++) {
            Arrays.fill(massAbundance[p], 0);
            Arrays.fill(nbrPrey[p], 0);

            resource[p] = init.resource;
            environment[p] = init.environment[p];
            for (int m = (p * comm.microsites); m < ((p + 1) * comm.microsites); m++)
                patch[m] = p;
            int[] posInds = Auxils.arraySample(init.N[p], Auxils.enumArray(p * comm.microsites, ((p + 1) * comm.microsites) - 1));
            for (int m : posInds) {
                indGtp = init.genotype[p];
                for (int l : evol.allGenes) {
                    genotype[m][l] = (byte) Math.round(Auxils.random.nextDouble() * 0.5 * (Auxils.random.nextBoolean() ? -1 : 1) + indGtp);
                }
                int s = Auxils.randIntCumProb(init.speciesCumProb[p]);
                newBody(m, s);
            }
        }
    }


    double calcPhenotype(int i) {
        return Auxils.arrayMean(genotype[i]) + (Auxils.gaussianSampler.sample() * evol.sigmaZ);
    }

    double calcFitness(double phenot, double env) {
        return Math.exp(-(Math.pow(phenot - env, 2)) / evol.divF);
    }

    void newBody(int i, int s) {
        int p = patch[i];
        species[i] = s;
        alive[i] = true;
        fitness[i] = 1;
        int mc = comm.massClassSpecies[s];
        double bm = comm.bodyMassClass[mc];
        mortality[i] = comm.d*Math.pow(bm, comm.dPow);
        massAbundance[p][mc]++;
        phenotype[i] = calcPhenotype(i);
        fitness[i] = calcFitness(phenotype[i], environment[p]);
        preyPos[p][mc][nbrPrey[p][mc]++] = i;
    }

    void deadBody(int i) {
        alive[i] = false;
        massAbundance[patch[i]][comm.massClassSpecies[species[i]]]--;
    }

    void changeEnvironment() {
        for (int p = 0; p < comm.nbrPatches; p++) {
            environment[p] = environment[p] + comm.envStep[esPos];
            adjustFitness(p);
        }
    }

    void adjustFitness(int p) {
        for (int m = (p * comm.microsites); m < ((p + 1) * comm.microsites); m++) {
            fitness[m] = calcFitness(phenotype[m], environment[p]);
        }
    }

    void uptake() {
        int mcPrey;
        double ingested, bm, bmPrey;
        double[] abundance = new double[comm.trophicLevels];
        for (int p = 0; p < comm.nbrPatches; p++) {
            Arrays.fill(consumption[p], 0);
            Arrays.fill(uptakePrey[p], 0);
            for (int mc = 0; mc < comm.trophicLevels; mc++)
                abundance[mc] = massAbundance[p][mc];
            for (int mc = 0; mc < comm.trophicLevels; mc++) {
                mcPrey = mc - 1;
                if (mcPrey == -1) {
                    ingested = (comm.uptakeMassClass[mc][0] * Math.pow(resource[p], comm.uptakeMassClass[mc][2])) /
                            (1 + comm.uptakeMassClass[mc][0] * comm.uptakeMassClass[mc][1] * Math.pow(resource[p], comm.uptakeMassClass[mc][2]));
                    resource[p] += (comm.inRate - resource[p] * comm.outRate) - ingested*abundance[mc];
                    resource[p] = Math.max(0, resource[p]);
//                        resource[p] += comm.R*(1 - resource[p]/comm.K) -  - ingested;
                    consumption[p][mc] += comm.resourceConversion*ingested;
                }
                else {
                    bm = comm.bodyMassClass[mc];
                    bmPrey = comm.bodyMassClass[mcPrey];
                    ingested = (comm.uptakeMassClass[mc][0] * Math.pow(bmPrey * abundance[mcPrey], comm.uptakeMassClass[mc][2])) /
                            (1 + comm.uptakeMassClass[mc][0] * comm.uptakeMassClass[mc][1] * Math.pow(bmPrey * abundance[mcPrey], comm.uptakeMassClass[mc][2]));
                    abundance[mcPrey] -= ingested*abundance[mc]/bmPrey;
                    consumption[p][mc] += comm.assimilationEff*ingested/bm;
                }
            }
            for (int mc = 0; mc < comm.trophicLevels; mc++)
                uptakePrey[p][mc] = Math.min(massAbundance[p][mc] - abundance[mc], massAbundance[p][mc]);
        }
    }

    void updatePrey() {
        int deadPrey;
        int[] posPrey;
        for (int p = 0; p < comm.nbrPatches; p++)
            for (int c = 0; c < comm.trophicLevels; c++) {
                deadPrey = (int) uptakePrey[p][c];
                posPrey = Auxils.arraySample(deadPrey, Arrays.copyOf(preyPos[p][c], nbrPrey[p][c]));
                for (int i = 0; i < deadPrey; i++)
                    deadBody(posPrey[i]);
                uptakePrey[p][c] = 0;
        }
    }

    void contributionAdults() {
        double contr;
        int p, p2, s, s2, mc, m;

        Arrays.fill(nbrEmpty, 0);
        for (p = 0; p < comm.nbrPatches; p++) {
            Arrays.fill(nbrPrey[p], 0);
            for (s = 0; s < comm.nbrSpecies; s++)
                Arrays.fill(fathersCumProb[p][s], 0);
        }
        for (int i = 0; i < totSites; i++) {
            p = patch[i];
            s = species[i];
            mc = comm.massClassSpecies[s];
            m = i%comm.microsites;
            if (alive[i] && (Auxils.random.nextDouble() > ((1 - mortality[i]) * fitness[i])))
                deadBody(i);
            if (alive[i]) {
                preyPos[p][mc][nbrPrey[p][mc]++] = i;
                contr = 1;
            }
            else {
                emptyPos[p][nbrEmpty[p]++] = i;
                contr = 0;
            }
            if (comm.repType.equals("SEXUAL")) {
                fathersCumProb[p][s][m] = contr;
                if (m > 0)
                    for(s2 = 0; s2 < comm.nbrSpecies; s2++)
                        fathersCumProb[p][s2][m] += fathersCumProb[p][s2][m-1];
            }
            contr *= consumption[p][mc];
            for (p2 = 0; p2 < comm.nbrPatches; p2++) {
                mothersCumProb[p2][i] = contr * comm.dispNeighbours[p2][p];
                if (i > 0)
                    mothersCumProb[p2][i] += mothersCumProb[p2][i-1];
            }
        }
        for (p = 0; p < comm.nbrPatches; p++) {
            nbrNewborns[p] = mothersCumProb[p][totSites-1];
            Auxils.arrayDiv(mothersCumProb[p], mothersCumProb[p][totSites-1]);
            if (comm.repType.equals("SEXUAL"))
                for (s = 0; s < comm.nbrSpecies; s++)
                    Auxils.arrayDiv(fathersCumProb[p][s], fathersCumProb[p][s][comm.microsites-1]);
        }
    }

    void reproduction() {
        int s, o, m, f, newborns, nbrSettled, patchMother;
        int[] posOffspring;
        for (int p = 0; p < comm.nbrPatches; p++) {
            newborns = (nbrNewborns[p] < 1) ? ((Auxils.random.nextDouble() <= nbrNewborns[p]) ? 1 : 0) : (int) nbrNewborns[p];
            nbrSettled = Math.min(newborns, nbrEmpty[p]);
            posOffspring = Auxils.arraySample(nbrSettled, Arrays.copyOf(emptyPos[p], nbrEmpty[p]));
            //sampling parents with replacement!
            //selfing allowed!
            for (int i = 0; i < nbrSettled; i++) {
                o = posOffspring[i];
                m = Auxils.randIntCumProb(mothersCumProb[p]);
                s = species[m];
                patchMother = patch[m];
               if (comm.repType.equals("SEXUAL")) {
                    f = patchMother*comm.microsites + Auxils.randIntCumProb(fathersCumProb[patchMother][s]);
                    inherit(o, m, f);
                } else
                    inherit(o, m);
                mutate(o);
                newBody(o, s);
            }
        }
    }

    /* inheritance for asexual reproduction (one parent) */
    void inherit(int posOffspring, int posParent) {
        System.arraycopy(genotype[posParent], 0, genotype[posOffspring], 0, 2 * evol.allLoci);
    }

    /* inheritance for sexual reproduction (two parent) */
    void inherit(int posOffspring, int posMother, int posFather) {
        for (int l = 0; l < evol.allLoci; l++) {
            genotype[posOffspring][evol.allMother[l]] = genotype[posMother][Auxils.random.nextBoolean() ? evol.allMother[l] : evol.allFather[l]];
            genotype[posOffspring][evol.allFather[l]] = genotype[posFather][Auxils.random.nextBoolean() ? evol.allMother[l] : evol.allFather[l]];
        }
    }

    void mutate(int posOffspring) {
        int k;
        int[] somMutLocs;

        k = Auxils.binomialSamplerSomatic.sample();
        if (k > 0) {
            CombinationSampler combinationSampler = new CombinationSampler(Auxils.random,evol.allLoci*2, k);
            somMutLocs = Auxils.arrayElements(evol.allGenes, combinationSampler.sample());
            for (int l : somMutLocs) {
                genotype[posOffspring][l] += (Auxils.random.nextBoolean() ? -1 : 1);
            }
        }
    }

    int foodwebSize() {
        int N = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                N++;
        return N;
    }

    double foodwebMass() {
        double M = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                M += comm.bodyMassClass[comm.massClassSpecies[species[i]]];
        return M;
    }

    int metaPopSize(int s) {
        int N = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i] && species[i] == s)
                N++;
        return N;
    }

    double metaPopMass(int s) {
        double M = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i] && species[i] == s)
                M++;
        M *= comm.bodyMassClass[comm.massClassSpecies[s]];
        return M;
    }

    int popSize(int p, int s) {
        int N = 0;
        for (int i = p*comm.microsites; i < ((p + 1) * comm.microsites); i++)
            if (alive[i] && species[i] == s)
                N++;
        return N;
    }

    double popMass(int p, int s) {
        double M = 0;
        for (int i = p*comm.microsites; i < ((p + 1) * comm.microsites); i++)
            if (alive[i] && species[i] == s)
                M++;
        M *= comm.bodyMassClass[comm.massClassSpecies[s]];
        return M;
    }

    double genotypeMean(int s) {
        double mean = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i] && species[i] == s)
                mean += Auxils.arrayMean(genotype[i]);
        mean /= metaPopSize(s);
        return mean;
    }

    double genotypeMean(int p, int s) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                mean += Auxils.arrayMean(genotype[i]);
        mean /= popSize(p, s);
        return mean;
    }

    double genotypeVar(int s) {
        double mean = genotypeMean(s);
        double var = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i] && species[i] == s)
                var += Math.pow(mean - Auxils.arrayMean(genotype[i]), 2);
        var /= metaPopSize(s);
        return var;
    }

    double genotypeVar(int p, int s) {
        double mean = genotypeMean(p, s);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                var += Math.pow(mean - Auxils.arrayMean(genotype[i]), 2);
        var /= popSize(p, s);
        return var;
    }

    double phenotypeMean(int s) {
        double mean = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i] && species[i] == s)
                mean += phenotype[i];
        mean /= metaPopSize(s);
        return mean;
    }

    double phenotypeMean(int p, int s) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                mean += phenotype[i];
        mean /= popSize(p, s);
        return mean;
    }

    double phenotypeVar(int s) {
        double mean = phenotypeMean(s);
        double var = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i] && species[i] == s)
                var += Math.pow(mean - phenotype[i], 2);
        var /= metaPopSize(s);
        return var;
    }

    double phenotypeVar(int p, int s) {
        double mean = phenotypeMean(p, s);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                var += Math.pow(mean - phenotype[i], 2);
        var /= popSize(p, s);
        return var;
    }

    double fitnessMean() {
        double mean = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                mean += fitness[i];
        mean /= foodwebSize();
        return mean;
    }

    double fitnessMean(int p, int s) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                mean += fitness[i];
        mean /= popSize(p, s);
        return mean;
    }

    double fitnessGeom(int p, int s) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                mean += Math.log(fitness[i]);
        mean /= popSize(p, s);
        return Math.exp(mean);
    }

    double fitnessVar(int p, int s) {
        double mean = fitnessMean(p, s);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                var += Math.pow(mean - fitness[i], 2);
        var /= popSize(p, s);
        return var;
    }

    double fitnessMin(int p, int s) {
        double min = 1;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++) {
            if (alive[i] && species[i] == s)
                if (fitness[i] < min)
                    min = fitness[i];
        }
        return min;
    }

    double fitnessMax(int p, int s) {
        double max = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++) {
            if (alive[i] && species[i] == s)
                if (fitness[i] > max)
                    max = fitness[i];
        }
        return max;
    }

    double loadMean(int p, int s) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                mean += 1 - fitness[i];
        mean /= popSize(p, s);
        return mean;
    }

    double loadVar(int p, int s) {
        double mean = loadMean(p, s);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i] && species[i] == s)
                var += Math.pow(mean - (1 - fitness[i]), 2);
        var /= popSize(p, s);
        return var;
    }

    double selectionDiff(int p, int s) {
        double mean = 0;
        double sum = 0;
        double fit;
        double fitMean = fitnessMean(p, s);
        double phenotypeSd = Math.sqrt(phenotypeVar(p, s));
        double SDiff;

        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++) {
            if (alive[i] && species[i] == s) {
                fit = fitness[i] / fitMean;
                mean += phenotype[i] * fit;
                sum += fit;
            }
        }
        mean /= sum;
        if (phenotypeSd == 0)
            SDiff = 0;
        else
            SDiff = Math.abs((mean - phenotypeMean(p, s))) / phenotypeSd;
        return SDiff;
    }
}


/* Ecological parameters/variables */
class Comm {
    String repType = "SEXUAL";
//    double minEnv = 0.2;
//    double maxEnv = 0.8;
    double[] envRangeX = {-1, 1};
    double[] envRangeY = {-1, 1};
    int microsites = 600;

    double inRate = 100;
    double outRate = 0.1;
    double resourceConversion = 10;
//    double resourceMass = 0.1;
//    double K = 10000;
//    double R = 100;


    double[] uptakePars = {1e-4, 0.4, 1.0};
    double iPow = 0.75;
    double assimilationEff = 0.7;

    int trophicLevels = 1;
    int nbrSpecies = 1;
    int[] massClassSpecies = new int[nbrSpecies];
    double d = 0.1;
    double dPow = -0.25;

    double[] bodyMassClass = new double[trophicLevels];
    double[][] uptakeMassClass = new double[trophicLevels][3];

    //    int gridSize = 2;
    int gridX = 2;
    int gridY = 2;
    String torusX = "NO";
    String torusY = "YES";
//    int nbrPatches = gridSize * gridSize;
    int nbrPatches = gridX * gridY;
    int[][] patchXY = new int[nbrPatches][2];
    double[] envStep = {0.01};
    double[] dispRate = {0.01};
    double rho = 1;

    double[][] neighbours = new double[nbrPatches][nbrPatches];
    double[][] dispNeighbours = new double[nbrPatches][nbrPatches];

    void init(Run run) {

        inRate *= run.dt;
        outRate *= run.dt;
//        R *= run.dt;
        d *= run.dt;
        Auxils.arrayMult(envStep, run.dt);
        uptakePars[0] *= run.dt;
        uptakePars[1] /= run.dt;

        bodyMassClass = new double[trophicLevels];
        uptakeMassClass = new double[trophicLevels][3];
        for (int c = 0; c < trophicLevels; c++) {
            bodyMassClass[c] = Math.pow(10, c);
            uptakeMassClass[c][0] = uptakePars[0] * Math.pow(bodyMassClass[c], iPow);
            uptakeMassClass[c][1] = uptakePars[1] * Math.pow(bodyMassClass[c], -iPow);
            uptakeMassClass[c][2] = uptakePars[2];
        }

        massClassSpecies = new int[nbrSpecies];
        for (int s = 0; s < nbrSpecies; s++) {
            massClassSpecies[s] = s%trophicLevels;
        }

//        nbrPatches = gridSize * gridSize;
        nbrPatches = gridX * gridY;
        patchXY = new int[nbrPatches][2];
        neighbours = new double[nbrPatches][nbrPatches];
        dispNeighbours = new double[nbrPatches][nbrPatches];
        calcPatchXY();
        calcDistNeighbours();
    }

    void calcPatchXY() {
        for (int x = 0; x < gridX; x++)
            for (int y = 0; y < gridY; y++) {
                patchXY[y * gridX + x][0] = x;
                patchXY[y * gridX + x][1] = y;
            }
    }

    void calcDistNeighbours() {
        double distX, distY;
        for (int x = 0; x < gridX; x++)
            for (int y = 0; y < gridY; y++)
                for (int x2 = 0; x2 < gridX; x2++)
                    for (int y2 = 0; y2 < gridY; y2++) {
                        distX = Math.abs(x - x2);
                        distX = torusX.equals("YES") ? Math.min(distX, gridX - distX) : distX;
                        distX = Math.pow(distX, 2);
                        distY = Math.abs(y - y2);
                        distY = torusY.equals("YES") ? Math.min(distY, gridY - distY) : distY;
                        distY = Math.pow(distY, 2);
                        neighbours[y * gridX + x][y2 * gridX + x2] = Math.sqrt(distX + distY);
                    }
//debug
//        System.out.println("    distNeighbours");
//        for (int p = 0; p < nbrPatches; p++)
//            System.out.println("      patch " + p + ":  " + Arrays.toString(neighbours[p]));
//

    }

    void calcDispNeighbours(int dr) {
        for (int i = 0; i < nbrPatches; i++) {
            for (int j = 0; j < nbrPatches; j++)
                dispNeighbours[i][j] = (i == j) ? 0 : (rho * Math.exp(-rho * neighbours[i][j]));
            double iSum = Auxils.arraySum(dispNeighbours[i]);
            for (int j = 0; j < nbrPatches; j++)
                dispNeighbours[i][j] = (i == j) ? (1 - dispRate[dr]) : (dispRate[dr] * dispNeighbours[i][j] / iSum);
        }

//debug
//        System.out.println("    dispNeighbours");
//        for (int p = 0; p < nbrPatches; p++)
//            System.out.println("      patch " + p + ":  " + Arrays.toString(dispNeighbours[p]));
//

    }
}


/* Evolution parameters/variables */
class Evol {
    double omegaE = 0.02;
    double divF = 1;
    int allLoci = 20;
    double mutationRate = 1e-4;
    double sigmaZ = 0.01;

    int[] allMother;
    int[] allFather;
    int[] allGenes;

    int longPos = 0;

    void init(Comm comm) {
        divF = 2 * Math.pow(omegaE, 2);

        allMother = new int[allLoci];
        allFather = new int[allLoci];
        allGenes = new int[2 * allLoci];

        /* all genes */
        for (int l = 0; l < allLoci; l++) {
            allMother[l] = l;
            allFather[l] = allMother[l] + allLoci;
        }
        allGenes = Auxils.arrayConcat(allMother, allFather);
    }
}


/* run parameters */
class Run {
    double dt = 0.1;
    int runs = 1;
    int timeSteps = 10000;
    int preChange = 1000;
    int postChange = 1000;
    int printSteps = 100;
    int saveSteps = 1000;
    String fileName = "output_evolution_of_sex.csv";
}


/* initialize simulation run */
class Init {
    double resource;
    double[] environment;
    double[][] speciesCumProb;
    int[] N;
    double[] genotype;

    public Init(Comm comm) {
        double dEnv;
        resource = (comm.inRate/comm.outRate)/2;
        environment = new double[comm.nbrPatches];
        speciesCumProb = new double[comm.nbrPatches][comm.nbrSpecies];
        N = new int[comm.nbrPatches];
        genotype = new double[comm.nbrPatches];

//        Arrays.fill(N, comm.microsites);
        Arrays.fill(N, comm.microsites/2);
//        Arrays.fill(speciesCumProb, 0);

        double stepX = comm.gridX == 1 ? 0 : (comm.envRangeX[1] - comm.envRangeX[0])/(comm.gridX-1);
        double stepY = comm.gridY == 1 ? 0 : (comm.envRangeY[1] - comm.envRangeY[0])/(comm.gridY-1);
        for (int p = 0; p < comm.nbrPatches; p++) {
            environment[p] = comm.envRangeX[0] + stepX*comm.patchXY[p][0] + comm.envRangeY[0] + stepY*comm.patchXY[p][1];
        }

        for (int p = 0; p < comm.nbrPatches; p++) {
//            species[p] = p%comm.nbrSpecies;
            for (int s = 0; s < comm.nbrSpecies; s++) {
                int pProb = s/(comm.nbrSpecies/comm.nbrPatches) == p ? 1 : 0;
                speciesCumProb[p][s] = pProb/comm.bodyMassClass[comm.massClassSpecies[s]];
            }
            Auxils.arrayCumSum(speciesCumProb[p]);
            Auxils.arrayDiv(speciesCumProb[p], speciesCumProb[p][comm.nbrSpecies-1]);
            genotype[p] = environment[p];
        }
    }
}


/* reading in parameter values from input file */
class Reader {
    static void readInput(String fileName, Comm comm, Evol evol, Run run) throws IOException {
        try (BufferedReader input = new BufferedReader(new FileReader(fileName))) {
            String line;
            String[] words;
            int size;
            while ((line = input.readLine()) != null) {
                words = line.trim().split("\\s+");
                switch (words[0]) {
//                    case "MINENV":
//                        comm.minEnv = Double.parseDouble(words[1]);
//                        break;
//                    case "MAXENV":
//                        comm.maxEnv = Double.parseDouble(words[1]);
//                        break;
                    case "ENVRANGEX":
                        for (int i = 0; i < 2; i++)
                            comm.envRangeX[i] = Double.parseDouble(words[1 + i]);
                        break;
                    case "ENVRANGEY":
                        for (int i = 0; i < 2; i++)
                            comm.envRangeY[i] = Double.parseDouble(words[1 + i]);
                        break;
                    case "MICROSITES":
                        comm.microsites = Integer.parseInt(words[1]);
                        break;
                    case "INRATE":
                        comm.inRate = Double.parseDouble(words[1]);
                        break;
                    case "NBRSPECIES":
                        comm.nbrSpecies = Integer.parseInt(words[1]);
                        break;
                    case "TROPHICLEVELS":
                        comm.trophicLevels = Integer.parseInt(words[1]);
                        break;
                    case "D":
                        comm.d = Double.parseDouble(words[1]);
                        break;
                    case "REPTYPE":
                        comm.repType = words[1];
                        break;
//                    case "GRIDSIZE":
//                        comm.gridSize = Integer.parseInt(words[1]);
//                        break;
                    case "GRIDX":
                        comm.gridX = Integer.parseInt(words[1]);
                        break;
                    case "GRIDY":
                        comm.gridY = Integer.parseInt(words[1]);
                        break;
                    case "ENVSTEP":
                        size = Integer.parseInt(words[1]);
                        comm.envStep = new double[size];
                        for (int i = 0; i < size; i++)
                            comm.envStep[i] = Double.parseDouble(words[2 + i]);
                        break;
                    case "M":
                        size = Integer.parseInt(words[1]);
                        comm.dispRate = new double[size];
                        for (int i = 0; i < size; i++)
                            comm.dispRate[i] = Double.parseDouble(words[2 + i]);
                        break;
                    case "RHO":
                        comm.rho = Double.parseDouble(words[1]);
                        break;

                    case "OMEGAE":
                        evol.omegaE = Double.parseDouble(words[1]);
                        break;
                    case "MU":
                        evol.mutationRate = Double.parseDouble(words[1]);
                        break;
                    case "SIGMAZ":
                        evol.sigmaZ = Double.parseDouble(words[1]);
                        break;

                    case "RUNS":
                        run.runs = Integer.parseInt(words[1]);
                        break;
                    case "TIMESTEPS":
                        run.timeSteps = Integer.parseInt(words[1]);
                        break;
                        case "PRECHANGE":
                        run.preChange = Integer.parseInt(words[1]);
                        break;
                    case "POSTCHANGE":
                        run.postChange = Integer.parseInt(words[1]);
                        break;
                    case "PRINTSTEPS":
                        run.printSteps = Integer.parseInt(words[1]);
                        break;
                    case "SAVESTEPS":
                        run.saveSteps = Integer.parseInt(words[1]);
                        break;
                    case "OUTPUT":
                        run.fileName = words[1];
                        break;
                }
            }
        }
    }
}


/* Auxiliary functions for array calculations */
class Auxils {
    //    static UniformRandomProvider random = RandomSource.create(RandomSource.MT_64);
//    static UniformRandomProvider random = RandomSource.create(RandomSource.JSF_64);
//    static UniformRandomProvider random = RandomSource.create(RandomSource.MSWS);
    static UniformRandomProvider random = RandomSource.create(RandomSource.XO_RO_SHI_RO_128_PP);

    static NormalizedGaussianSampler gaussianSampler = ZigguratNormalizedGaussianSampler.of(random);
    static SharedStateDiscreteSampler binomialSamplerSomatic;

    static void init(Evol evol) {
        binomialSamplerSomatic = Binomial.of(random, evol.allLoci*2, evol.mutationRate);
    }

    static void arrayShuffle(int[] array) {
        int index, temp;
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    static void arrayShuffle(double[] array) {
        int index;
        double temp;
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    static int[] arraySample(int n, int[] array) {
        int[] tempArr = array.clone();
        arrayShuffle(tempArr);
        return Arrays.copyOf(tempArr, n);
    }

    static double[] arraySample(int n, double[] array) {
        double[] tempArr = array.clone();
        arrayShuffle(tempArr);
        return Arrays.copyOf(tempArr, n);
    }

    //sampling with or without replacement
    static int[] arraySampleProb(int n, int[] array, double[] probs, boolean repl) {
        int pos;
        int[] newElements;
        int[] newArr = new int[n];
        double[] cumProbs = probs.clone();
        arrayCumSum(cumProbs);
        arrayDiv(cumProbs, cumProbs[cumProbs.length - 1]);
        for (int i = 0; i < n; i++) {
            pos = Arrays.binarySearch(cumProbs, random.nextDouble());
            pos = (pos >= 0) ? pos : (-pos - 1);
            newArr[i] = array[pos];
            if (!repl) {
                newElements = arrayConcat(enumArray(0, pos - 1), enumArray(pos + 1, array.length - 1));
                array = arrayElements(array, newElements);
                probs = arrayElements(probs, newElements);
                cumProbs = probs.clone();
                arrayCumSum(cumProbs);
                arrayDiv(cumProbs, cumProbs[cumProbs.length - 1]);
            }
        }
        return newArr;
    }

    static int randIntProb(int end, double[] probs) {
        int val;
        double[] cumProbs = Arrays.copyOf(probs, end);
        Auxils.arrayCumSum(cumProbs);
        Auxils.arrayDiv(cumProbs, cumProbs[cumProbs.length - 1]);
        val = Arrays.binarySearch(cumProbs, random.nextDouble());
        return (val >= 0) ? val : (-val - 1);
    }

    static int randIntCumProb(double[] cumProbs) {
        int val;
        val = Arrays.binarySearch(cumProbs, random.nextDouble());
        return (val >= 0) ? val : (-val - 1);
    }

    static int[] enumArray(int from, int to) {
        int[] newArr = new int[to - from + 1];
        for (int i = 0; i < newArr.length; i++)
            newArr[i] = from++;
        return newArr;
    }

    static int[] arrayElements(int[] array, int[] pos) {
        int[] newArr = new int[pos.length];
        for (int i = 0; i < newArr.length; i++)
            newArr[i] = array[pos[i]];
        return newArr;
    }

    static byte[] arrayElements(byte[] array, int[] pos) {
        byte[] newArr = new byte[pos.length];
        for (int i = 0; i < newArr.length; i++)
            newArr[i] = array[pos[i]];
        return newArr;
    }

    static boolean[] arrayElements(boolean[] array, int[] pos) {
        boolean[] newArr = new boolean[pos.length];
        for (int i = 0; i < newArr.length; i++)
            newArr[i] = array[pos[i]];
        return newArr;
    }

    static double[] arrayElements(double[] array, int[] pos) {
        double[] newArr = new double[pos.length];
        for (int i = 0; i < newArr.length; i++)
            newArr[i] = array[pos[i]];
        return newArr;
    }

    static int[] arrayAntiElements(int[] array, int[] pos) {
        java.util.Arrays.sort(pos);
        int j = 0;
        int k = 0;
        int[] newArr = new int[array.length - pos.length];
        for (int i = 0; i < array.length; i++)
            if(i == j)
                j++;
            else
                newArr[k++] = array[i];
        return newArr;
    }

    static double arrayMean(int[] array) {
        double mean = 0;
        for (int value : array) mean += value;
        mean /= array.length;
        return mean;
    }

    static double arrayMean(byte[] array) {
        double mean = 0;
        for (byte value : array) mean += value;
        mean /= array.length;
        return mean;
    }

    static double arrayMean(boolean[] array) {
        double mean = 0;
        for (boolean b : array)
            if (b)
                mean++;
        mean /= array.length;
        return mean;
    }

    static double arrayMean(double[] array) {
        double mean = 0;
        for (double v : array) mean += v;
        mean /= array.length;
        return mean;
    }

    static double arrayMean(int[] array, int end) {
        double mean = 0;
        for (int i = 0; i < end; i++)
            mean += array[i];
        mean /= end;
        return mean;
    }

    static double arrayMean(boolean[] array, int end) {
        double mean = 0;
        for (int i = 0; i < end; i++)
            if (array[i])
                mean++;
        mean /= end;
        return mean;
    }

    static double arrayMean(double[] array, int end) {
        double mean = 0;
        for (int i = 0; i < end; i++)
            mean += array[i];
        mean /= end;
        return mean;
    }

    static int arrayMax(int[] array) {
        int max = array[0];
        for (int i = 1; i < array.length; i++)
            if (array[i] > max)
                max = array[i];
        return max;
    }

    static double arrayMax(double[] array) {
        double max = array[0];
        for (int i = 1; i < array.length; i++)
            if (array[i] > max)
                max = array[i];
        return max;
    }

    static int arrayMin(int[] array) {
        int min = array[0];
        for (int i = 1; i < array.length; i++)
            if (array[i] < min)
                min = array[i];
        return min;
    }

    static double arrayMin(double[] array) {
        double min = array[0];
        for (int i = 1; i < array.length; i++)
            if (array[i] < min)
                min = array[i];
        return min;
    }

    static int arraySum(int[] array) {
        int sum = 0;
        for (int value : array) sum += value;
        return sum;
    }

    static int arraySum(boolean[] array) {
        int sum = 0;
        for (boolean b : array)
            if (b)
                sum++;
        return sum;
    }

    static double arraySum(double[] array) {
        double sum = 0;
        for (double v : array) sum += v;
        return sum;
    }

    static int arraySum(int[] array, int end) {
        int sum = 0;
        for (int i = 0; i < end; i++)
            sum += array[i];
        return sum;
    }

    static int arraySum(boolean[] array, int end) {
        int sum = 0;
        for (int i = 0; i < end; i++)
            if (array[i])
                sum++;
        return sum;
    }

    static double arraySum(double[] array, int end) {
        double sum = 0;
        for (int i = 0; i < end; i++)
            sum += array[i];
        return sum;
    }

    static void arrayCumSum(int[] array) {
        for (int i = 1; i < array.length; i++)
            array[i] += array[i - 1];
    }

    static void arrayCumSum(double[] array) {
        for (int i = 1; i < array.length; i++)
            array[i] += array[i - 1];
    }

    static void arrayAdd(int[] array, int a) {
        for (int i = 0; i < array.length; i++)
            array[i] += a;
    }

    static void arrayAdd(double[] array, double a) {
        for (int i = 0; i < array.length; i++)
            array[i] += a;
    }

    static void arrayMult(int[] array, int a) {
        for (int i = 0; i < array.length; i++)
            array[i] *= a;
    }

    static void arrayMult(double[] array, double a) {
        for (int i = 0; i < array.length; i++)
            array[i] *= a;
    }

    static void arrayDiv(int[] array, int a) {
        for (int i = 0; i < array.length; i++)
            array[i] /= a;
    }

    static void arrayDiv(double[] array, double a) {
        for (int i = 0; i < array.length; i++)
            array[i] /= a;
    }

    static int[] arrayConcat(int[] first, int[] second) {
        int[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }

    static double[] arrayConcat(double[] first, double[] second) {
        double[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }

    static double adjustToRange(double val, double min, double max) {
        double range = max - min;
        int quot = (int) Math.floor((val - max) / range);
        double rem = mod((val - max), range);
        int minAdd = mod(quot, 2);
        int maxAdd = 1 - minAdd;
        return minAdd * (min + rem) + maxAdd * (max - rem);
    }

    static int mod(int x, int y) {
        int result = x % y;
        return result < 0 ? result + y : result;
    }

    static double mod(double x, double y) {
        double result = x % y;
        return result < 0 ? result + y : result;
    }

    static double mod(int x, double y) {
        double result = x % y;
        return result < 0 ? result + y : result;
    }

    static double mod(double x, int y) {
        double result = x % y;
        return result < 0 ? result + y : result;
    }
}


