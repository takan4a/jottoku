package jp.kobe_u.sugar.encoder;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StreamTokenizer;
import java.util.BitSet;
import java.util.Iterator;

import jp.kobe_u.sugar.csp.BooleanVariable;
import jp.kobe_u.sugar.csp.CSP;
import jp.kobe_u.sugar.csp.IntegerVariable;

public class Models implements Iterator<CSP> {
    private CSP csp;
    private StreamTokenizer st;
    private BitSet satValues;

    public Models(CSP csp, String outFileName) throws IOException {
        this.csp = csp;
        BufferedReader rd = new BufferedReader(new FileReader(outFileName));
        st = new StreamTokenizer(rd);
        st.eolIsSignificant(true);
        satValues = nextModel();
    }

    public BitSet nextModel() {
        BitSet satValues = null;
        boolean eom = false;
        try {
            if (st == null)
                return null;
            while (! eom) {
                st.nextToken();
                if (st.ttype == StreamTokenizer.TT_EOF) {
                    st = null;
                    return satValues;
                }
                switch (st.ttype) {
                case StreamTokenizer.TT_EOL:
                    break;
                case StreamTokenizer.TT_WORD:
                    if (st.sval.equals("s") || st.sval.equals("o")) {
                        do {
                            st.nextToken();
                        } while (st.ttype != StreamTokenizer.TT_EOL);
                    } else if (st.sval.equals("c")) {
                        do {
                            st.nextToken();
                        } while (st.ttype != StreamTokenizer.TT_EOL);
                    } else if (st.sval.equals("v")) {
                        if (satValues == null)
                            satValues = new BitSet();
                        do {
                            st.nextToken();
                            int value = (int) st.nval;
                            int i = Math.abs(value);
                            if (i == 0) {
                                eom = true;
                            } else { 
                                satValues.set(i, value > 0);
                            }
                        } while (st.ttype != StreamTokenizer.TT_EOL);
                    } else {
                        // throw new SugarException("Unknown output " + st.sval);
                    }
                    break;
                case StreamTokenizer.TT_NUMBER:
                    if (satValues == null)
                        satValues = new BitSet();
                    int value = (int) st.nval;
                    int i = Math.abs(value);
                    if (i == 0) {
                        eom = true;
                    } else { 
                        satValues.set(i, value > 0);
                    }
                    break;
                default:
                    // throw new SugarException("Unknown output " + st.sval);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return satValues;
    }

    @Override
    public boolean hasNext() {
        return satValues != null;
    }

    @Override
    public CSP next() {
        if (satValues != null) {
            for (IntegerVariable v : csp.getIntegerVariables()) {
                v.decode(satValues);
            }
            for (BooleanVariable v : csp.getBooleanVariables()) {
                v.decode(satValues);
            }
        }
        satValues = nextModel();
        return csp;
    }

}
