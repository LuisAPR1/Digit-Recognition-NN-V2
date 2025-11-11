import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Utilitário para carregar o dataset MNIST no formato IDX original.
 * Os ficheiros podem ser descarregados através do script download_mnist.py
 * e devem permanecer descomprimidos na pasta data/.
 */
public class MnistLoader {

    public static final int IMAGE_MAGIC = 2051;
    public static final int LABEL_MAGIC = 2049;

    public static class MnistDataset {
        private final double[][] images;
        private final int[] labels;

        public MnistDataset(double[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }

        public double[][] getImages() {
            return images;
        }

        public int[] getLabels() {
            return labels;
        }
    }

    public static MnistDataset load(String imagesPath, String labelsPath, int limit) throws IOException {
        double[][] images = loadImages(imagesPath, limit);
        int[] labels = loadLabels(labelsPath, limit);

        if (images.length != labels.length) {
            throw new IOException("Número de imagens (" + images.length + ") diferente do número de labels (" + labels.length + ")");
        }

        return new MnistDataset(images, labels);
    }

    public static double[][] loadImages(String path, int limit) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int magic = dis.readInt();
            if (magic != IMAGE_MAGIC) {
                throw new IOException("Ficheiro de imagens inválido: magic number " + magic);
            }

            int totalImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();
            int pixelsPerImage = rows * cols;
            int imagesToRead = limit > 0 ? Math.min(limit, totalImages) : totalImages;
            double[][] images = new double[imagesToRead][pixelsPerImage];
            byte[] buffer = new byte[pixelsPerImage];

            final double mean = 0.1307;
            final double std = 0.3081;

            for (int i = 0; i < imagesToRead; i++) {
                dis.readFully(buffer);
                for (int j = 0; j < pixelsPerImage; j++) {
                    int unsigned = buffer[j] & 0xFF;
                    double normalized = (unsigned / 255.0 - mean) / std;
                    images[i][j] = normalized;
                }
            }
            return images;
        }
    }

    public static int[] loadLabels(String path, int limit) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int magic = dis.readInt();
            if (magic != LABEL_MAGIC) {
                throw new IOException("Ficheiro de labels inválido: magic number " + magic);
            }

            int totalLabels = dis.readInt();
            int labelsToRead = limit > 0 ? Math.min(limit, totalLabels) : totalLabels;
            int[] labels = new int[labelsToRead];

            for (int i = 0; i < labelsToRead; i++) {
                labels[i] = dis.readUnsignedByte();
            }
            return labels;
        }
    }
}
