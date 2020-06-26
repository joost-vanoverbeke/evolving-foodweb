

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

                            System.out.format("run = %d; dims = %d; traits = %d; disp = %.4f; step = %.4f%n",
                                    (r + 1), comm.envDims, comm.traits, comm.dispRate[dr], comm.envStep[es]);

                            comm.init();
                            evol.init(comm);
                            Auxils.init(evol);
                            Init init = new Init(comm);

                            sites = new Sites(comm, evol, init, es, dr);

                            System.out.format("  time = %d; metacommunity N = %d; fit = %f%n", 0, sites.metaPopSize(), sites.fitnessMean());
                            logResults(0, streamOut, r, es, dr);

                            for (int t = 0; t < run.timeSteps; t++) {
                                sites.changeEnvironment();
                                sites.contributionAdults();
                                sites.reproduction();

                                if (t == 0 || ((t + 1) % run.printSteps) == 0) {
                                    System.out.format("  time = %d; metacommunity N = %d; fit = %f%n", (t + 1), sites.metaPopSize(), sites.fitnessMean());
                                }
                                if (t == 0 || ((t + 1) % run.saveSteps) == 0) {
                                    logResults(t+1, streamOut, r, es, dr);
                                }
                            }
                        }

            long endTime = System.currentTimeMillis();
            System.out.println("EvolMetac took " + (endTime - startTime) +
                    " milliseconds.");
        }
    }

    static void logTitles(PrintWriter out) {
        out.print("gridsize;patches;p_e_change;e_step;m;rho;dims;sigma_e;microsites;d;b;traits;traitLoci;sigma_z;mu;omega_e;"
                + "run;time;patch;N;fitness_mean;fitness_var;fitness_geom;load_mean;load_var;S_mean;S_var");
        for (int tr = 0; tr < comm.traits; tr++)
            out.format(";dim_tr%d;e_dim_tr%d;genotype_mean_tr%d;genotype_var_tr%d;phenotype_mean_tr%d;phenotype_var_tr%d;fitness_mean_tr%d;fitness_var_tr%d;"
                            + "genotype_meta_var_tr%d;phenotype_meta_var_tr%d",
                    tr + 1, tr + 1, tr + 1, tr + 1, tr + 1, tr + 1, tr + 1, tr + 1, tr + 1, tr + 1);
        out.println("");
    }

    static void logResults(int t, PrintWriter out, int r, int es, int dr) {
        for (int p = 0; p < comm.nbrPatches; p++) {
            out.format("%d;%d;%f;%f;%f;%f;%d;%f;%d;%f;%f;%d;%d;%f;%f;%f",
                    comm.gridSize, comm.nbrPatches, comm.pChange, comm.envStep[es], comm.dispRate[dr], comm.rho, comm.envDims, comm.sigmaE, comm.microsites, comm.d, comm.b, comm.traits, evol.traitLoci, evol.sigmaZ, evol.mutationRate, evol.omegaE);
            out.format(";%d;%d;%d",
                    r + 1, t, p + 1);
            out.format(";%d;%f;%f;%f;%f;%f;%f;%f",
                    sites.popSize(p), sites.fitnessMean(p), sites.fitnessVar(p), sites.fitnessGeom(p), sites.loadMean(p), sites.loadVar(p), sites.selectionDiff(p), sites.selectionDiffVar(p));
            for (int tr = 0; tr < comm.traits; tr++)
                out.format(";%d;%f;%f;%f;%f;%f;%f;%f;%f;%f",
                        sites.comm.traitDim[tr] + 1, sites.environment[p][sites.comm.traitDim[tr]], sites.genotypeMean(p, tr), sites.genotypeVar(p, tr), sites.phenotypeMean(p, tr), sites.phenotypeVar(p, tr), sites.traitFitnessMean(p, tr), sites.traitFitnessVar(p, tr),
                        sites.genotypeVar(tr), sites.phenotypeVar(tr));
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
    boolean[] alive;
    double[][] traitPhenotype;
    double[][] traitFitness;
    double[] fitness;

    byte[][] genotype;

    double[][] environment;

    double[] nbrNewborns;
    int[] nbrEmpty;

    int[][] emptyPos;
    double[][] mothersCumProb;
    double[][] fathersCumProb;

    public Sites(Comm cmm, Evol evl, Init init, int es, int dr) {
        comm = cmm;
        evol = evl;
        esPos = es;
        drPos = dr;

        comm.calcDispNeighbours(drPos);

        totSites = comm.nbrPatches * comm.microsites;

        patch = new int[totSites];
        alive = new boolean[totSites];
        traitPhenotype = new double[totSites][comm.traits];
        traitFitness = new double[totSites][comm.traits];
        fitness = new double[totSites];
        genotype = new byte[totSites][2 * evol.allLoci];

        environment = new double[comm.nbrPatches][comm.envDims];

        nbrNewborns = new double[comm.nbrPatches];
        nbrEmpty = new int[comm.nbrPatches];

        emptyPos = new int[comm.nbrPatches][comm.microsites];
        mothersCumProb = new double[comm.nbrPatches][totSites];
        fathersCumProb = new double[comm.nbrPatches][comm.microsites];

        double indGtp;
        Arrays.fill(alive, false);

        for (int p = 0; p < comm.nbrPatches; p++) {
            if (comm.envDims >= 0) System.arraycopy(init.environment[p], 0, environment[p], 0, comm.envDims);
            for (int m = (p * comm.microsites); m < ((p + 1) * comm.microsites); m++)
                patch[m] = p;
            int[] posInds = Auxils.arraySample(init.N[p], Auxils.enumArray(p * comm.microsites, ((p + 1) * comm.microsites) - 1));
            for (int m : posInds) {
                alive[m] = true;
                fitness[m] = 1;
                for (int tr = 0; tr < comm.traits; tr++) {
                    traitFitness[m][tr] = 1;
                    indGtp = init.genotype[p][tr];
                    for (int l : evol.traitGenes[tr]) {
                        genotype[m][l] = (byte) Math.round(Auxils.random.nextDouble() * 0.5 * (Auxils.random.nextBoolean() ? -1 : 1) + indGtp);
                    }
                    traitPhenotype[m][tr] = calcPhenotype(m, tr);
                    traitFitness[m][tr] = calcFitness(traitPhenotype[m][tr], environment[p][comm.traitDim[tr]]);
                    fitness[m] *= traitFitness[m][tr];
                }
            }
        }
    }


    double calcPhenotype(int i, int tr) {
        return Auxils.arrayMean(Auxils.arrayElements(genotype[i], evol.traitGenes[tr])) + (Auxils.gaussianSampler.sample() * evol.sigmaZ);
    }

    double calcFitness(double phenot, double env) {
        return Math.exp(-(Math.pow(phenot - env, 2)) / evol.divF);
    }

    void changeEnvironment() {
        boolean globalEnv = comm.envType.equals("GLOBAL");
        boolean globalChange;
        double globalStep = 0;
        double step;

        for (int d = 0; d < comm.envDims; d++) {
            globalChange = globalEnv && (Auxils.random.nextDouble() <= comm.pChange);
            if (globalChange) globalStep = comm.envStep[esPos] * (Auxils.random.nextBoolean() ? -1 : 1);
            for (int p = 0; p < comm.nbrPatches; p++) {
                if (globalEnv ? globalChange : (Auxils.random.nextDouble() <= comm.pChange)) {
                    step = globalEnv ? globalStep : (comm.envStep[esPos] * (Auxils.random.nextBoolean() ? -1 : 1));
                    environment[p][d] = environment[p][d] + step;
                    environment[p][d] = Auxils.adjustToRange(environment[p][d], comm.minEnv, comm.maxEnv);
                    adjustFitness(p, d);
                }
            }
        }
    }

    void adjustFitness(int p, int d) {
        double oldFit;
        for (int m = (p * comm.microsites); m < ((p + 1) * comm.microsites); m++) {
            oldFit = fitness[m];
            if (oldFit == 0)
                fitness[m] = 1;
            for (int tr = 0; tr < comm.traits; tr++) {
                if ((oldFit != 0) && (comm.traitDim[tr] == d))
                    fitness[m] /= traitFitness[m][tr];
                if ((oldFit == 0) || (comm.traitDim[tr] == d)) {
                    traitFitness[m][tr] = calcFitness(traitPhenotype[m][tr], environment[p][comm.traitDim[tr]]);
                    fitness[m] *= traitFitness[m][tr];
                }
            }
        }
    }

    void contributionAdults() {
        double contr;
        int p, m;

        Arrays.fill(nbrEmpty, 0);

        for (int i = 0; i < totSites; i++) {
            p = patch[i];
            m = i%comm.microsites;
            if(alive[i])
                alive[i] = Auxils.random.nextDouble() <= (1 - comm.d);
//            alive[i] = alive[i] ? (Auxils.random.nextDouble() <= ((1 - comm.d)*fitness[i])) : alive[i];
            if (!alive[i]) {
//                nbrEmpty[p]++;
                emptyPos[p][nbrEmpty[p]++] = i;
            }
//            emptyPos[p][m] = alive[i] ? 0 : 1;
//            if (m > 0)
//                emptyPos[p][m] += emptyPos[p][m-1];
            contr = alive[i] ? comm.b : 0;
            contr *= fitness[i];
            if (comm.repType.equals("SEXUAL")) {
                fathersCumProb[p][m] = contr;
                if (m > 0)
                fathersCumProb[p][m] += fathersCumProb[p][m-1];
            }
            for (int p2 = 0; p2 < comm.nbrPatches; p2++) {
                mothersCumProb[p2][i] = contr * comm.dispNeighbours[p2][p];
                if (i > 0)
                    mothersCumProb[p2][i] += mothersCumProb[p2][i-1];
            }
        }
        for (p = 0; p < comm.nbrPatches; p++) {
            nbrNewborns[p] = mothersCumProb[p][totSites-1];
//            nbrEmpty[p] = (int) emptyPos[p][comm.microsites-1];
            Auxils.arrayDiv(mothersCumProb[p], mothersCumProb[p][totSites-1]);
//            Auxils.arrayDiv(emptyPos[p], emptyPos[p][comm.microsites-1]);
            if (comm.repType.equals("SEXUAL"))
                Auxils.arrayDiv(fathersCumProb[p], fathersCumProb[p][comm.microsites-1]);
        }
    }

    void reproduction() {
        int o, m, f, newborns, nbrSettled, patchMother;
        int[] posOffspring;
        for (int p = 0; p < comm.nbrPatches; p++) {
            newborns = (nbrNewborns[p] < 1) ? ((Auxils.random.nextDouble() <= nbrNewborns[p]) ? 1 : 0) : (int) nbrNewborns[p];
            nbrSettled = Math.min(newborns, nbrEmpty[p]);
            posOffspring = Auxils.arraySample(nbrSettled, Arrays.copyOf(emptyPos[p], nbrEmpty[p]));

//debug
//            System.out.println(" patch " + p);
//            System.out.println("   nbrNewborns " + nbrNewborns[p]);
//            System.out.println("   nbrEmpty " + nbrEmpty[p]);
//            System.out.println("   newborns " + newborns);
//            System.out.println("   settled " + nbrSettled);

            //sampling parents with replacement!
            //selfing allowed!
            for (int i = 0; i < nbrSettled; i++) {
                o = posOffspring[i];
                m = Auxils.randIntCumProb(mothersCumProb[p]);
                patchMother = patch[m];

//debug
//                System.out.println("     offspr " + i);
//                System.out.println("       pos o " + o);
//                System.out.println("       alive o " + alive[o]);
//                System.out.println("       pos m " + m);
//                System.out.println("       alive m " + alive[m]);


                if (comm.repType.equals("SEXUAL")) {
                    f = patchMother*comm.microsites + Auxils.randIntCumProb(fathersCumProb[patchMother]);
                    inherit(o, m, f);
                } else
                    inherit(o, m);
                mutate(o);
                alive[o] = true;
                fitness[o] = 1;
                for (int tr = 0; tr < comm.traits; tr++) {
                    traitPhenotype[o][tr] = calcPhenotype(o, tr);
                    traitFitness[o][tr] = calcFitness(traitPhenotype[o][tr], environment[p][comm.traitDim[tr]]);
                    fitness[o] *= traitFitness[o][tr];
                }
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
            CombinationSampler combinationSampler = new CombinationSampler(Auxils.random,evol.traitLoci*2, k);
            somMutLocs = Auxils.arrayElements(evol.allGenes, combinationSampler.sample());
            for (int l : somMutLocs) {
                genotype[posOffspring][l] += (Auxils.random.nextBoolean() ? -1 : 1);
            }
        }
    }

    int metaPopSize() {
        int N = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                N++;
        return N;
    }

    int popSize(int p) {
        int N = 0;
        for (int i = p*comm.microsites; i < ((p + 1) * comm.microsites); i++)
            if (alive[i])
                N++;
        return N;
    }

    double genotypeMean(int t) {
        double mean = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                mean += Auxils.arrayMean(Auxils.arrayElements(genotype[i], evol.traitGenes[t]));
        mean /= metaPopSize();
        return mean;
    }

    double genotypeMean(int p, int t) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                mean += Auxils.arrayMean(Auxils.arrayElements(genotype[i], evol.traitGenes[t]));
        mean /= popSize(p);
        return mean;
    }

    double genotypeVar(int t) {
        double mean = genotypeMean(t);
        double var = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                var += Math.pow(mean - Auxils.arrayMean(Auxils.arrayElements(genotype[i], evol.traitGenes[t])), 2);
        var /= metaPopSize();
        return var;
    }

    double genotypeVar(int p, int t) {
        double mean = genotypeMean(p, t);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                var += Math.pow(mean - Auxils.arrayMean(Auxils.arrayElements(genotype[i], evol.traitGenes[t])), 2);
        var /= popSize(p);
        return var;
    }

    double phenotypeMean(int t) {
        double mean = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                mean += traitPhenotype[i][t];
        mean /= metaPopSize();
        return mean;
    }

    double phenotypeMean(int p, int t) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                mean += traitPhenotype[i][t];
        mean /= popSize(p);
        return mean;
    }

    double phenotypeVar(int t) {
        double mean = phenotypeMean(t);
        double var = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                var += Math.pow(mean - traitPhenotype[i][t], 2);
        var /= metaPopSize();
        return var;
    }

    double phenotypeVar(int p, int t) {
        double mean = phenotypeMean(p, t);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                var += Math.pow(mean - traitPhenotype[i][t], 2);
        var /= popSize(p);
        return var;
    }

    double traitFitnessMean(int p, int t) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                mean += traitFitness[i][t];
        mean /= popSize(p);
        return mean;
    }

    double traitFitnessVar(int p, int t) {
        double mean = traitFitnessMean(p, t);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                var += Math.pow(mean - traitFitness[i][t], 2);
        var /= popSize(p);
        return var;
    }

    double fitnessMean() {
        double mean = 0;
        for (int i = 0; i < totSites; i++)
            if (alive[i])
                mean += fitness[i];
        mean /= metaPopSize();
        return mean;
    }

    double fitnessMean(int p) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                mean += fitness[i];
        mean /= popSize(p);
        return mean;
    }

    double fitnessGeom(int p) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                mean += Math.log(fitness[i]);
        mean /= popSize(p);
        return Math.exp(mean);
    }

    double fitnessVar(int p) {
        double mean = fitnessMean(p);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                var += Math.pow(mean - fitness[i], 2);
        var /= popSize(p);
        return var;
    }

    double fitnessMin(int p) {
        double min = 1;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++) {
            if (alive[i])
                if (fitness[i] < min)
                    min = fitness[i];
        }
        return min;
    }

    double fitnessMax(int p) {
        double max = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++) {
            if (alive[i])
                if (fitness[i] > max)
                    max = fitness[i];
        }
        return max;
    }

    double loadMean(int p) {
        double mean = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                mean += 1 - fitness[i];
        mean /= popSize(p);
        return mean;
    }

    double loadVar(int p) {
        double mean = loadMean(p);
        double var = 0;
        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++)
            if (alive[i])
                var += Math.pow(mean - (1 - fitness[i]), 2);
        var /= popSize(p);
        return var;
    }

    double selectionDiff(int p, int t) {
        double mean = 0;
        double sum = 0;
        double fit;
        double fitMean = fitnessMean(p);
        double phenotypeSd = Math.sqrt(phenotypeVar(p, t));
        double SDiff;

        for (int i = p * comm.microsites; i < (p + 1) * comm.microsites; i++) {
            if (alive[i]) {
                fit = fitness[i] / fitMean;
                mean += traitPhenotype[i][t] * fit;
                sum += fit;
            }
        }
        mean /= sum;
        if (phenotypeSd == 0)
            SDiff = 0;
        else
            SDiff = Math.abs((mean - phenotypeMean(p, t))) / phenotypeSd;
        return SDiff;
    }

    double selectionDiff(int p) {
        double mean = 0;
        for (int t = 0; t < comm.traits; t++)
            mean += selectionDiff(p, t);
        mean /= comm.traits;
        return mean;
    }

    double selectionDiffVar(int p) {
        double mean = selectionDiff(p);
        double var = 0;
        for (int t = 0; t < comm.traits; t++)
            var += Math.pow(mean - selectionDiff(p, t), 2);
        var /= popSize(p);
        return var;
    }
}


/* Ecological parameters/variables */
class Comm {
    String envType = "GLOBAL";
    String repType = "SEXUAL";
    int envDims = 1;
    int traits = 2;
    double minEnv = 0.2;
    double maxEnv = 0.8;
    double sigmaE = 0.0;
    int microsites = 600;
    double d = 0.1;
    double b = 1;

    int gridSize = 2;
    int nbrPatches = gridSize * gridSize;
    double pChange = 0.1;
    double[] envStep = {0.01};
    double[] dispRate = {0.01};
    double rho = 1;

    int[] traitDim;

    double[][] neighbours = new double[nbrPatches][nbrPatches];
    double[][] dispNeighbours = new double[nbrPatches][nbrPatches];

    void init() {
        nbrPatches = gridSize * gridSize;
        neighbours = new double[nbrPatches][nbrPatches];
        dispNeighbours = new double[nbrPatches][nbrPatches];
        calcDistNeighbours();

        traitDim = new int[traits];
        int dim = 0;
        for (int tr = 0; tr < traits; tr++) {
            traitDim[tr] = dim++;
            if (dim == envDims)
                dim = 0;
        }
    }

    void calcDistNeighbours() {
        for (int i = 0; i < gridSize; i++)
            for (int j = 0; j < gridSize; j++)
                for (int i2 = 0; i2 < gridSize; i2++)
                    for (int j2 = 0; j2 < gridSize; j2++) {
                        double dist = Math.sqrt(Math.pow(Math.min(Math.abs(i - i2), gridSize - Math.abs(i - i2)), 2) + Math.pow(Math.min(Math.abs(j - j2), gridSize - Math.abs(j - j2)), 2));
                        neighbours[j * gridSize + i][j2 * gridSize + i2] = dist;
                    }
    }

    void calcDispNeighbours(int dr) {
        for (int i = 0; i < nbrPatches; i++) {
            for (int j = 0; j < nbrPatches; j++)
                dispNeighbours[i][j] = (i == j) ? 0 : (rho * Math.exp(-rho * neighbours[i][j]));
            double iSum = Auxils.arraySum(dispNeighbours[i]);
            for (int j = 0; j < nbrPatches; j++)
                dispNeighbours[i][j] = (i == j) ? (1 - dispRate[dr]) : (dispRate[dr] * dispNeighbours[i][j] / iSum);
        }
    }
}


/* Evolution parameters/variables */
class Evol {
    double omegaE = 0.02;
    double divF = 1;
    int allLoci = 20;
    int traitLoci = 20;
    int lociPerTrait = traitLoci;
    double mutationRate = 1e-4;
    double sigmaZ = 0.01;

    int[] allMother;
    int[] allFather;
    int[] allGenes;
    int[][] traitMother;
    int[][] traitFather;
    int[][] traitGenes;

    int longPos = 0;

    void init(Comm comm) {
        divF = 2 * Math.pow(Math.sqrt(comm.traits) * omegaE, 2);

        allLoci = traitLoci;
        lociPerTrait = traitLoci / comm.traits;

        allMother = new int[allLoci];
        allFather = new int[allLoci];
        allGenes = new int[2 * allLoci];
        traitMother = new int[comm.traits][lociPerTrait];
        traitFather = new int[comm.traits][lociPerTrait];
        traitGenes = new int[comm.traits][2 * lociPerTrait];

        /* somatic genes */
        for (int tr = 0; tr < comm.traits; tr++) {
            for (int l = 0; l < lociPerTrait; l++) {
                longPos = l + (tr * lociPerTrait);
                traitMother[tr][l] = longPos;
                traitFather[tr][l] = traitMother[tr][l] + allLoci;
            }
            traitGenes[tr] = Auxils.arrayConcat(traitMother[tr], traitFather[tr]);
        }

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
    int runs = 1;
    int timeSteps = 10000;
    int printSteps = 100;
    int saveSteps = 1000;
    String fileName = "output_evolution_of_sex.csv";
}


/* initialize simulation run */
class Init {
    double pSex;

    double[][] environment;
    int[] N;
    double[][] genotype;

    public Init(Comm comm) {
        double dEnv;
        environment = new double[comm.nbrPatches][comm.envDims];
        N = new int[comm.nbrPatches];
        genotype = new double[comm.nbrPatches][comm.traits];

        Arrays.fill(N, comm.microsites);

        if (comm.envType.equals("GLOBAL")) {
            for (int d = 0; d < comm.envDims; d++) {
                dEnv = comm.minEnv + (Auxils.random.nextDouble() * (comm.maxEnv - comm.minEnv));
                for (int p = 0; p < comm.nbrPatches; p++)
                    environment[p][d] = dEnv;
            }
        } else {
            for (int p = 0; p < comm.nbrPatches; p++) {
                for (int d = 0; d < comm.envDims; d++) {
                    environment[p][d] = comm.minEnv + (Auxils.random.nextDouble() * (comm.maxEnv - comm.minEnv));
                }
            }
        }

        for (int p = 0; p < comm.nbrPatches; p++) {
            for (int tr = 0; tr < comm.traits; tr++) {
                genotype[p][tr] = environment[p][comm.traitDim[tr]];
            }
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
                    case "ENVDIMS":
                        comm.envDims = Integer.parseInt(words[1]);
                        break;
                    case "TRAITS":
                        comm.traits = Integer.parseInt(words[1]);
                        break;
                    case "MINENV":
                        comm.minEnv = Double.parseDouble(words[1]);
                        break;
                    case "MAXENV":
                        comm.maxEnv = Double.parseDouble(words[1]);
                        break;
                    case "SIGMAE":
                        comm.sigmaE = Double.parseDouble(words[1]);
                        break;
                    case "MICROSITES":
                        comm.microsites = Integer.parseInt(words[1]);
                        break;
                    case "D":
                        comm.d = Double.parseDouble(words[1]);
                        break;
                    case "B":
                        comm.b = Double.parseDouble(words[1]);
                        break;
                    case "REPTYPE":
                        comm.repType = words[1];
                        break;
                    case "GRIDSIZE":
                        comm.gridSize = Integer.parseInt(words[1]);
                        break;
                    case "ENVTYPE":
                        comm.envType = words[1];
                        break;
                    case "PCHANGE":
                        comm.pChange = Double.parseDouble(words[1]);
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
                    case "TRAITLOCI":
                        evol.traitLoci = Integer.parseInt(words[1]);
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
        binomialSamplerSomatic = Binomial.of(random, evol.traitLoci*2, evol.mutationRate);
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


