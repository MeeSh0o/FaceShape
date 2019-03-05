package com.xs.util.image.other;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import com.sun.image.codec.jpeg.JPEGImageEncoder;
import com.sun.image.codec.jpeg.JPEGCodec;
public class TextToImage {
	 /** 文本文件  */
    private File textFile;
    /** 图片文件 */
    private File imageFile;
    /** 图片 */
    private BufferedImage image;
    /** 图片宽度  */
    private final int IMAGE_WIDTH = 1723;
    /** 图片高度 */
    private final int IMAGE_HEIGHT = 1300;
    /** 图片类型  */
    private final int IMAGE_TYPE = BufferedImage.TYPE_INT_RGB;
     
    /**
     * 构造函数
     * @param textFile 文本文件
     * @param imageFile 图片文件
     */
    public TextToImage(File textFile,File imageFile){
        this.textFile = textFile;
        this.imageFile = imageFile;
        this.image = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_TYPE);
    }
     
    /**
     * 将文本文件里文字，写入到图片中保存
     * @return boolean  true，写入成功；false，写入失败
     */
    public boolean convert() {
         
        //读取文本文件
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(textFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
         
        //获取图像上下文
        Graphics2D g = createGraphics(image);
        String line;
        //图片中文本行高
        final int Y_LINEHEIGHT = 30;
        int lineNum = 1;
        try {
            while((line = reader.readLine()) != null){
            	String context = line;
            	int lengthRight = line.replaceAll("\\s+$","").length();//去掉右边空格的长度
            	System.out.println(line);
            	int lengthTrim = line.trim().length();//去掉空格的长度
            	int X = (lengthRight-lengthTrim)*7;//列
//                g.drawString(line+"\r\n",X,lineNum * Y_LINEHEIGHT);
//                g.drawString(line+"\r\n",0,lineNum * Y_LINEHEIGHT);
            	int x = getWatermarkLength(line, g);
//            	int y = 2*getWatermarkLength(line, g);
            	 g.drawString(context,X-375,lineNum * Y_LINEHEIGHT);
                lineNum++;
            }
            g.dispose();
             
            //保存为jpg图片
            FileOutputStream fos = new FileOutputStream(imageFile);
            JPEGImageEncoder encoder = JPEGCodec.createJPEGEncoder(fos);
            encoder.encode(image);
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }
     
    /**
     * 获取到图像上下文
     * @param image 图片
     * @return Graphics
     */
    private Graphics2D createGraphics(BufferedImage image){
    	Graphics2D g = image.createGraphics();
        g.setColor(Color.WHITE); //设置背景色
        g.fillRect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);//绘制背景
        g.setColor(Color.BLACK); //设置前景色
        g.setFont(new Font("微软雅黑", Font.PLAIN, 11)); //设置字体
        return g;
    }
    public static void main(String[] args) {
        TextToImage convert = new TextToImage(new File("G:/image.txt"), new File("G:/image.jpg"));
        boolean success = convert.convert();
        System.out.println("文本转图片：" + (success ? "成功" : "失败"));
    }
    public static int getWatermarkLength(String waterMarkContent, Graphics2D g) {  
        return g.getFontMetrics(g.getFont()).charsWidth(waterMarkContent.toCharArray(), 0, waterMarkContent.length());  
    }  
}
