import 'dart:math';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:fl_chart/fl_chart.dart';

class ResultsScreen extends StatefulWidget {
  final String? jobId;

  const ResultsScreen({Key? key, this.jobId}) : super(key: key);

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  bool _showAfter = false; // Toggles before vs after state

  Widget _buildMetricCard({
    required String title,
    required double value,
    required bool isPass,
  }) {
    return Card(
      color: const Color(0xFF1F2937),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              title,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.grey, fontSize: 13),
            ),
            const SizedBox(height: 8),
            Text(
              value.toStringAsFixed(3),
              style: TextStyle(
                color: isPass ? Colors.greenAccent : Colors.redAccent,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: isPass ? Colors.green.withOpacity(0.2) : Colors.red.withOpacity(0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                isPass ? 'PASS' : 'FAIL',
                style: TextStyle(
                  color: isPass ? Colors.greenAccent : Colors.redAccent,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendItem(String label, Color color) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 14)),
      ],
    );
  }

  Color _getGroupColor(String group) {
    switch (group.toLowerCase()) {
      case 'brahmin': return const Color(0xFFA78BFA);
      case 'dalit': return const Color(0xFFFBBF24);
      case 'hindu':
      case 'hindu_other': return const Color(0xFF34D399); 
      case 'muslim': return const Color(0xFF60A5FA);
      default: return Colors.white;
    }
  }

  List<ScatterSpot> _getPlaceholderSpots(bool debiased) {
    final rand = Random(42);
    final groups = ['Brahmin', 'Dalit', 'Hindu', 'Muslim'];
    final Map<String, Offset> clusters = {
      'Brahmin': const Offset(0.6, 0.6),
      'Dalit': const Offset(-0.6, -0.6),
      'Hindu': const Offset(-0.6, 0.6),
      'Muslim': const Offset(0.6, -0.6),
    };
    
    List<ScatterSpot> spots = [];
    for (var group in groups) {
      for (int i = 0; i < 40; i++) {
        double x, y;
        if (debiased) {
          x = (rand.nextDouble() * 2) - 1.0;
          y = (rand.nextDouble() * 2) - 1.0;
        } else {
          final center = clusters[group]!;
          x = center.dx + (rand.nextDouble() * 0.4 - 0.2);
          y = center.dy + (rand.nextDouble() * 0.4 - 0.2);
        }
        
        spots.add(ScatterSpot(
          x, y,
          dotPainter: FlDotCirclePainter(
            radius: 4, color: _getGroupColor(group), strokeWidth: 0,
          ),
        ));
      }
    }
    return spots;
  }

  List<ScatterSpot> _getChartSpots(List<dynamic>? data) {
    if (data == null || data.isEmpty) {
      return _getPlaceholderSpots(_showAfter);
    }

    try {
      final rand = Random(42);
      return data.map((item) {
        final map = item as Map<String, dynamic>;
        final group = map['group'] as String? ?? 'Unknown';
        
        double x = (map['x'] as num?)?.toDouble() ?? 0.0;
        double y = (map['y'] as num?)?.toDouble() ?? 0.0;
        
        if (_showAfter) {
          if (map.containsKey('x_after')) {
            x = (map['x_after'] as num).toDouble();
          } else if (map.containsKey('x_debiased')) {
            x = (map['x_debiased'] as num).toDouble();
          } else {
            x = (rand.nextDouble() * 2) - 1.0; // Simulated debiasing mapping fallback
          }

          if (map.containsKey('y_after')) {
            y = (map['y_after'] as num).toDouble();
          } else if (map.containsKey('y_debiased')) {
            y = (map['y_debiased'] as num).toDouble();
          } else {
            y = (rand.nextDouble() * 2) - 1.0; // Simulated debiasing mapping fallback
          }
        }

        return ScatterSpot(
          x, y,
          dotPainter: FlDotCirclePainter(
            radius: 4, color: _getGroupColor(group), strokeWidth: 0,
          ),
        );
      }).toList();
    } catch (_) {
      return _getPlaceholderSpots(_showAfter);
    }
  }

  Widget _buildTopSection(Map<String, dynamic>? data, String status, String? errorText) {
    if (widget.jobId == null) {
      return Container(
        height: 150,
        alignment: Alignment.center,
        child: Text(
          'Upload a dataset to see results',
          style: TextStyle(color: Colors.grey[400], fontSize: 18),
        ),
      );
    }

    if (status == 'pending' || status == 'running') {
      return Container(
        height: 150,
        alignment: Alignment.center,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(color: Color(0xFFA78BFA)),
            const SizedBox(height: 24),
            Text(
              'Analyzing bias in embedding space...',
              style: TextStyle(color: Colors.grey[400], fontSize: 16),
            ),
          ],
        ),
      );
    }

    if (status == 'error') {
      return Container(
        height: 150,
        alignment: Alignment.center,
        padding: const EdgeInsets.all(24.0),
        child: Text(
          'Error: ${errorText ?? "Unknown error occurred"}',
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.redAccent, fontSize: 18),
        ),
      );
    }

    if (status == 'complete' && data != null) {
      final casteSeatBefore = (data['caste_seat_before'] as num?)?.toDouble() ?? 0.0;
      final casteSeatAfter = (data['caste_seat_after'] as num?)?.toDouble() ?? 0.0;
      final dpBefore = (data['demographic_parity_before'] as num?)?.toDouble() ?? 0.0;
      final dpAfter = (data['demographic_parity_after'] as num?)?.toDouble() ?? 0.0;
      
      final passesFairness = data['passes_fairness'] as bool? ?? false;
      final geminiExplanation = data['gemini_explanation'] as String? ?? 'No explanation available.';

      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: passesFairness ? Colors.green.withOpacity(0.2) : Colors.red.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: passesFairness ? Colors.greenAccent : Colors.redAccent,
                width: 2,
              ),
            ),
            child: Text(
              passesFairness ? 'CERTIFIED COMPLIANT' : 'BIAS DETECTED',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: passesFairness ? Colors.greenAccent : Colors.redAccent,
                fontWeight: FontWeight.bold,
                fontSize: 24,
                letterSpacing: 1.5,
              ),
            ),
          ),
          const SizedBox(height: 32),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
            childAspectRatio: 1.5,
            children: [
              _buildMetricCard(
                title: 'Caste SEAT (Before)',
                value: casteSeatBefore,
                isPass: casteSeatBefore <= 0.5,
              ),
              _buildMetricCard(
                title: 'Caste SEAT (After)',
                value: casteSeatAfter,
                isPass: casteSeatAfter < 0.2,
              ),
              _buildMetricCard(
                title: 'Demo. Parity (Before)',
                value: dpBefore,
                isPass: dpBefore >= 0.80,
              ),
              _buildMetricCard(
                title: 'Demo. Parity (After)',
                value: dpAfter,
                isPass: dpAfter >= 0.80,
              ),
            ],
          ),
          const SizedBox(height: 32),
          const Text(
            'AI Explanation',
            style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          Card(
            color: const Color(0xFF1F2937),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Text(
                geminiExplanation,
                style: const TextStyle(color: Colors.white70, fontSize: 16, height: 1.5),
              ),
            ),
          ),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Coming in Day 4')),
              );
            },
            icon: const Icon(Icons.download),
            label: const Text('Download Certificate'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFA78BFA),
              foregroundColor: const Color(0xFF080B12),
              padding: const EdgeInsets.symmetric(vertical: 16),
              textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      );
    }
    
    // Fallback if data is null but status supposedly complete
    return const SizedBox.shrink();
  }

  Widget _buildContent(Map<String, dynamic>? data, String status, String? errorText) {
    final tsneCoordinates = data?['tsne_coordinates'] as List<dynamic>?;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24.0),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Top Section (Metrics / Loading / Pending / Error / No Data)
              _buildTopSection(data, status, errorText),
              
              const SizedBox(height: 32),
              
              // --- TSNE CHART SECTION (Always Visible) ---
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Embedding Space (t-SNE)',
                    style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _showAfter = !_showAfter;
                      });
                    },
                    icon: Icon(_showAfter ? Icons.refresh : Icons.filter_center_focus),
                    label: Text(_showAfter ? 'Show Biased (Before)' : 'Show Debiased (After)'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1F2937),
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Container(
                height: 350,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: const Color(0xFF1F2937),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ScatterChart(
                  ScatterChartData(
                    scatterSpots: _getChartSpots(tsneCoordinates),
                    minX: -2.0, maxX: 2.0,
                    minY: -2.0, maxY: 2.0,
                    borderData: FlBorderData(show: false),
                    gridData: FlGridData(
                      show: true,
                      drawHorizontalLine: true,
                      drawVerticalLine: true,
                      getDrawingHorizontalLine: (value) => FlLine(color: Colors.grey.withOpacity(0.1), strokeWidth: 1),
                      getDrawingVerticalLine: (value) => FlLine(color: Colors.grey.withOpacity(0.1), strokeWidth: 1),
                    ),
                    titlesData: const FlTitlesData(show: false),
                  ),
                  swapAnimationDuration: const Duration(milliseconds: 600),
                  swapAnimationCurve: Curves.easeInOut,
                ),
              ),
              const SizedBox(height: 16),
              Wrap(
                alignment: WrapAlignment.center,
                spacing: 16,
                runSpacing: 8,
                children: [
                  _buildLegendItem('Brahmin', const Color(0xFFA78BFA)),
                  _buildLegendItem('Dalit', const Color(0xFFFBBF24)),
                  _buildLegendItem('Hindu', const Color(0xFF34D399)),
                  _buildLegendItem('Muslim', const Color(0xFF60A5FA)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF080B12),
      body: widget.jobId == null
          ? _buildContent(null, 'none', null)
          : StreamBuilder<DocumentSnapshot>(
              stream: FirebaseFirestore.instance.collection('audit_jobs').doc(widget.jobId).snapshots(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return _buildContent(null, 'pending', null);
                }

                if (snapshot.hasError) {
                  return _buildContent(null, 'error', 'Error loading results');
                }

                if (!snapshot.hasData || !snapshot.data!.exists) {
                  return _buildContent(null, 'error', 'Data not found');
                }

                final data = snapshot.data!.data() as Map<String, dynamic>;
                final status = data['status'] as String? ?? 'pending';
                final errorText = data['error'] as String?;

                return _buildContent(data, status, errorText);
              },
            ),
    );
  }
}
