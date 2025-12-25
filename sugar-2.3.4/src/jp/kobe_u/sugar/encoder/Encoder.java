package jp.kobe_u.sugar.encoder;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.StreamTokenizer;
import java.util.BitSet;

import jp.kobe_u.sugar.SugarConstants;
import jp.kobe_u.sugar.SugarException;
import jp.kobe_u.sugar.SugarMain;
import jp.kobe_u.sugar.csp.BooleanVariable;
import jp.kobe_u.sugar.csp.CSP;
import jp.kobe_u.sugar.csp.CSP.Objective;
import jp.kobe_u.sugar.csp.IntegerVariable;

/**
 * Encoder encodes CSP into SAT.
 * @see CSP 
 * @author Naoyuki Tamura (tamura@kobe-u.ac.jp)
 */
public class Encoder extends OrderEncoder {
    public Encoder(CSP csp) {
        super(csp, null);
    }

    public void commit() throws SugarException {
        problem.commit();
    }
    
    public void cancel() throws SugarException {
        problem.cancel();
    }
    
    public int getSatVariablesCount() {
        return problem.variablesCount;
    }

    public int getSatClausesCount() {
        return problem.clausesCount;
    }

    public long getSatFileSize() {
        return problem.fileSize;
    }
    
    public void encode(Problem problem) throws SugarException {
        this.problem = problem;
        if (csp.getGroups() > 0) {
            problem.setGroups(csp.getGroups(), csp.getTopWeight());
        }
        encode();
    }

    public void encode(String satFileName) throws SugarException {
        encode(new FileProblem(satFileName));
    }

    public void encode(String satFileName, boolean incremental) throws SugarException {
        if (incremental)
            throw new SugarException("incremental is not supported");
        encode(satFileName);
    }

    /*
    public void solveSAT() throws IOException, InterruptedException {
        File outFile = new File(outFileName);
        if (outFile.exists()) {
            outFile.delete();
        }
        String[] command = { satSolverName, satFileName, outFileName };
        SugarMain.log(satSolverName + " " + satFileName + " " + outFileName);
        Process process = Runtime.getRuntime().exec(command);
        BufferedReader stdout = new BufferedReader(
                new InputStreamReader(process.getInputStream()));
        BufferedReader stderr = new BufferedReader(
                new InputStreamReader(process.getErrorStream()));
        while (true) {
            String line = stderr.readLine();
            if (line == null)
                break;
            SugarMain.log(line);
        }
        stderr.close();
        while (true) {
            String line = stdout.readLine();
            if (line == null)
                break;
            SugarMain.log(line);
        }
        stdout.close();
        process.waitFor();
    }
    */

    public String summary() {
        return problem.summary();
    }
}
