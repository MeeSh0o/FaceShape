package com.xs.util.image.other;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.font.FontRenderContext;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import javax.imageio.ImageIO;

import com.xs.util.image.other.CalculateCharArea.CharArea;
/**
 * 图片转ascii字符文本
 * @author 小帅丶
 *
 */
public class Image2ASCII {
	private static char[] gradations = new char[256];
	private final static int WIDTH = 100;
	private static String path = "G:/";
	private static String filenameTemp ="image.txt";
	/**
	 * 对图片进行缩放，缩放至设定好的width
	 * @param source_bi
	 * @return BufferedImage
	 * @throws IOException
	 */
	private static BufferedImage zoom(BufferedImage source_bi) throws IOException {
		int height = source_bi.getHeight();
		int width = source_bi.getWidth();
		BufferedImage bi = new BufferedImage(WIDTH, height * WIDTH / width,
				BufferedImage.TYPE_BYTE_GRAY);

		Graphics g = bi.getGraphics();
		g.drawImage(source_bi, 0, 0, WIDTH, height * WIDTH / width, null);
		g.dispose();

		File file = new File("e://final.jpg");

		ImageIO.write(bi, "jpg", file);

		return bi;
	}

	/**
	 * 将计算出来的字符数据转化为256色阶中对应位置
	 * 
	 * @param arrays
	 */
	public static void hashChars2Arrays(CharArea[] arrays) {
		int min = arrays[0].pixNum;
		int max = arrays[arrays.length - 1].pixNum;
		int gap = max - min;
		for (CharArea charArea : arrays) {
			gradations[(int) ((charArea.pixNum - min) * 255 / gap)] = charArea.c;
		}
		char c = gradations[0];
		for (int i = 1; i < 255; i++) {
			if (gradations[i] == '\0')
				gradations[i] = c;
			else
				c = gradations[i];
		}
		for (char s : gradations) {
			System.out.print(s + " ");
		}
	}

	public static void print(int height, int width, BufferedImage bi) {
		int b = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int pixel = bi.getRGB(j, i); // 下面三行代码将一个数字转换为RGB数字
				b = (pixel & 0xff);
				System.out.printf("%2s", gradations[b]);
			}
			System.out.print('\n');
		}

	}

	public static void main(String[] args) throws IOException {
		// 引入文件
//		InputStream imageis = new FileInputStream(new File("G:/body2.jpg"));
//		BufferedImage bi = zoom(TransTool.trans2GrayImBuffer(imageis));
//		int height = bi.getHeight();
//		int width = bi.getWidth();
//		CharArea[] arrays = CalculateCharArea.getArrays();
//		hashChars2Arrays(arrays);
//		gradations[0] = ' ';
//		print(height, width, bi);
//		String result = BitmapConvert("G:/body2.jpg");
//		System.out.println(result);
		createAsciiPic("G:/woman.jpg");
	}

	public static String BitmapConvert(String imgpath) {
		StringBuffer _sb = new StringBuffer();
		File imgfile = new File(imgpath);
		char[] charset = { 'M', '8', '0', 'V', '1', 'i', ':', '*', '|', '.',
				' ' };
		try {
			BufferedImage buff = ImageIO.read(imgfile);

			int bitmapH = buff.getHeight();
			int bitmapW = buff.getWidth();

			for (int y = 0; y < bitmapH; y++) {
				for (int x = 0; x < bitmapW; x++) {
					int rgb = buff.getRGB(x, y);
					Color c = new Color(rgb);
					int cc = (c.getRed() + c.getGreen() + c.getBlue()) / 3;
					_sb.append(charset[(int) ((cc * 10 - 1) / 255)]);
				}
				_sb.append("\r\n");
			}
		} catch (Exception e) {
		}
		return _sb.toString();
	}
	public static void createAsciiPic(String path) {
		final String base = "@#&$%*o!;.";// 字符串由复杂到简单
//		final String base = "#8XOHLTI)i=+;:,. ";// 字符串由复杂到简单  
//		final String base = "M8V|:. ";
		StringBuffer buffer = new StringBuffer();
		String result = "";
		try {
			BufferedImage image = ImageHelper.resize(ImageIO.read(new File(path)),150,150);
			for (int y = 0; y < image.getHeight(); y += 2) {
				for (int x = 0; x < image.getWidth(); x++) {
					final int pixel = image.getRGB(x, y);
					final int r = (pixel & 0xff0000) >> 16, g = (pixel & 0xff00) >> 8, b = pixel & 0xff;
					final float gray = 0.299f * r + 0.578f * g + 0.114f * b;
					final int index = Math.round(gray * (base.length() + 1) / 255);
//					buffer.append(index >= base.length() ? " " : String.valueOf(base.charAt(index)));
					result += index >= base.length() ? " " : String.valueOf(base.charAt(index));
				}
//				buffer.append("\n");
//			    result =buffer.toString();
				result += "\r\n";
			}
			writeTxtFile(result);
			System.out.println(result);
		} catch (final IOException e) {
			e.printStackTrace();
		}
		
	}
	public static void Txt2Pic(String txt){
        int width = 250;      
        int height = 250;      
        File file = new File("G:/image.jpg");      
          
        Font font = new Font("Serif", Font.BOLD, 10);      
       //创建一个画布  
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);      
       //获取画布的画笔  
        Graphics2D g2 = (Graphics2D)bi.getGraphics();      
       //开始绘图  
        g2.setBackground(Color.WHITE);      
        g2.clearRect(0, 0, width, height);      
        g2.fillRect(0, 0, 100, 10);  
        g2.fillRect(0, 10, 100, 10);  
        g2.setPaint(Color.black);  
         
        FontRenderContext context = g2.getFontRenderContext();      
        Rectangle2D bounds = font.getStringBounds(txt, context);      
        double x = (width - bounds.getWidth()) / 2;      
        double y = (height - bounds.getHeight()) / 2;      
        double ascent = -bounds.getY();      
        double baseY = y + ascent;      
       //绘制字符串  
        g2.drawString(txt, (int)x, (int)baseY);   

        try {  
           //将生成的图片保存为jpg格式的文件。ImageIO支持jpg、png、gif等格式  
           ImageIO.write(bi, "jpg", file);  
       } catch (IOException e) {  
           System.out.println("生成图片出错........");  
           e.printStackTrace();  
       }      
	}
	public static boolean writeTxtFile(String newStr) throws IOException {
		// 先读取原有文件内容，然后进行写入操作
		boolean flag = false;
		String filein = newStr + "\r\n";
		String temp = "";
 
		FileInputStream fis = null;
		InputStreamReader isr = null;
		BufferedReader br = null;
 
		FileOutputStream fos = null;
		PrintWriter pw = null;
		try {
			// 文件路径
			File file = new File(path+filenameTemp);
			if(!file.exists()){
				file.createNewFile();
			}
			// 将文件读入输入流
			fis = new FileInputStream(file);
			isr = new InputStreamReader(fis);
			br = new BufferedReader(isr);
			StringBuffer buf = new StringBuffer();
 
			// 保存该文件原有的内容
			for (int j = 1; (temp = br.readLine()) != null; j++) {
				buf = buf.append(temp);
				buf = buf.append("\n");
				// System.getProperty("line.separator")
				// 行与行之间的分隔符 相当于“\n”
//				buf = buf.append(System.getProperty("line.separator"));
			}
			buf.append(filein);
 
			fos = new FileOutputStream(file);
			pw = new PrintWriter(fos);
			pw.write(buf.toString().toCharArray());
			pw.flush();
			flag = true;
		} catch (IOException e1) {
			// TODO 自动生成 catch 块
			throw e1;
		} finally {
			if (pw != null) {
				pw.close();
			}
			if (fos != null) {
				fos.close();
			}
			if (br != null) {
				br.close();
			}
			if (isr != null) {
				isr.close();
			}
			if (fis != null) {
				fis.close();
			}
		}
		return flag;
	}
}
