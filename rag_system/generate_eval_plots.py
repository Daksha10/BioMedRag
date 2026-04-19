import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Ensure the output directory exists
output_dir = "/Users/daksha/Projects/medical_RAG_system-main/evaluation_results/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Retrieval Performance (Bar Chart: Precision/Recall)
methods = ['BM25', 'DPR', 'Hybrid']
precision = [0.35, 0.42, 0.48]
recall = [0.58, 0.65, 0.72]

fig1 = go.Figure(data=[
    go.Bar(name='Precision@10', x=methods, y=precision, marker_color='#238636'),
    go.Bar(name='Recall@10', x=methods, y=recall, marker_color='#1f6feb')
])
fig1.update_layout(
    title='Retrieval Performance Comparison',
    xaxis_title='Retrieval Method',
    yaxis_title='Metric Score',
    barmode='group',
    template='plotly_dark'
)
fig1.write_image(os.path.join(output_dir, "retrieval_performance.png"))

# 2. Latency vs. Accuracy (Scatter Plot)
# Accuracy here is represented by Precision for demonstration
latency = [0.5, 3.5, 4.2] # seconds
accuracy = [0.35, 0.42, 0.48]

fig2 = px.scatter(
    x=latency, y=accuracy, text=methods,
    labels={'x': 'Latency (seconds)', 'y': 'Precision@10'},
    title='Efficiency vs. Accuracy Trade-off',
    template='plotly_dark'
)
fig2.update_traces(textposition='top center', marker=dict(size=15, color='#f85149'))
fig2.write_image(os.path.join(output_dir, "latency_vs_accuracy.png"))

# 3. Qualitative Comparison (Radar Chart)
categories = ['Speed', 'Semantic Relevance', 'Keyword Coverage', 
              'Medical Accuracy', 'Scalability']

fig3 = go.Figure()

fig3.add_trace(go.Scatterpolar(
      r=[10, 4, 9, 6, 10],
      theta=categories,
      fill='toself',
      name='BM25',
      line_color='#8b949e'
))
fig3.add_trace(go.Scatterpolar(
      r=[6, 9, 5, 8, 7],
      theta=categories,
      fill='toself',
      name='DPR',
      line_color='#1f6feb'
))
fig3.add_trace(go.Scatterpolar(
      r=[5, 9.5, 9.5, 9, 8],
      theta=categories,
      fill='toself',
      name='Hybrid',
      line_color='#238636'
))

fig3.update_layout(
  polar=dict(
    radialaxis=dict(visible=True, range=[0, 10])
  ),
  showlegend=True,
  title='Method Multi-Dimensional Comparison',
  template='plotly_dark'
)
fig3.write_image(os.path.join(output_dir, "radar_comparison.png"))

print(f"✅ All charts generated successfully in {output_dir}")
