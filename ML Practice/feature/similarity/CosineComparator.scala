public class CosineComparator extends AbstractComparator {
    @Override
    public <T> String compare(T t1, T t2) throws Exception {
        super.compare(t1, t2);

        Cosine cosine = new Cosine();
        return "" + cosine.similarity(t1.toString(), t2.toString());
    }
}
