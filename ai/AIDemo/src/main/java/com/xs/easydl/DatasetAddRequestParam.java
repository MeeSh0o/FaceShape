package com.xs.easydl;

import java.util.List;

public class DatasetAddRequestParam {
	private String type;
	private int dataset_id;
	private boolean appendLabel;
	private String entity_content;
	private String entity_name;
	private List<Labels> labels;
	
	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public int getDataset_id() {
		return dataset_id;
	}
	public void setDataset_id(int dataset_id) {
		this.dataset_id = dataset_id;
	}
	public boolean isAppendLabel() {
		return appendLabel;
	}
	public void setAppendLabel(boolean appendLabel) {
		this.appendLabel = appendLabel;
	}

	public String getEntity_content() {
		return entity_content;
	}
	public void setEntity_content(String entity_content) {
		this.entity_content = entity_content;
	}

	public String getEntity_name() {
		return entity_name;
	}

	public void setEntity_name(String entity_name) {
		this.entity_name = entity_name;
	}

	public List<Labels> getLabels() {
		return labels;
	}

	public void setLabels(List<Labels> labels) {
		this.labels = labels;
	}

	public static class Labels{
		private String label_name;
		private Integer left;
		private Integer top;
		private Integer width;
		private Integer height;
		public String getLabel_name() {
			return label_name;
		}
		public void setLabel_name(String label_name) {
			this.label_name = label_name;
		}
		public Integer getLeft() {
			return left;
		}
		public void setLeft(Integer left) {
			this.left = left;
		}
		public Integer getTop() {
			return top;
		}
		public void setTop(Integer top) {
			this.top = top;
		}
		public Integer getWidth() {
			return width;
		}
		public void setWidth(Integer width) {
			this.width = width;
		}
		public Integer getHeight() {
			return height;
		}
		public void setHeight(Integer height) {
			this.height = height;
		}
	}
}
