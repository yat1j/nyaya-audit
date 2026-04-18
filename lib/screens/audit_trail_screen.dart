import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class AuditTrailScreen extends StatelessWidget {
  final String? jobId;

  const AuditTrailScreen({Key? key, this.jobId}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (jobId == null) {
      return Scaffold(
        backgroundColor: const Color(0xFF080B12),
        body: Center(
          child: Text(
            'Upload a dataset to see audit trail',
            style: TextStyle(color: Colors.grey[400], fontSize: 18),
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFF080B12),
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance
            .collection('audit_jobs')
            .doc(jobId)
            .collection('retroactive_results')
            .snapshots(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator(color: Color(0xFFA78BFA)));
          }

          if (snapshot.hasError) {
            return const Center(child: Text('Error loading results', style: TextStyle(color: Colors.redAccent)));
          }

          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return const Center(child: Text('No audit trail data available.', style: TextStyle(color: Colors.white)));
          }

          final docs = snapshot.data!.docs;
          int totalApplicants = docs.length;
          int changedOutcomes = 0;

          for (var doc in docs) {
            final data = doc.data() as Map<String, dynamic>;
            final changed = data['changed']?.toString().toLowerCase() == 'yes' || data['changed'] == true;
            if (changed) changedOutcomes++;
          }

          return SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1000),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Summary Line
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1F2937),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          Column(
                            children: [
                              const Text('Total Applicants', style: TextStyle(color: Colors.grey, fontSize: 14)),
                              const SizedBox(height: 8),
                              Text('$totalApplicants', style: const TextStyle(color: Colors.white, fontSize: 32, fontWeight: FontWeight.bold)),
                            ],
                          ),
                          Column(
                            children: [
                              const Text('Outcomes Changed', style: TextStyle(color: Colors.grey, fontSize: 14)),
                              const SizedBox(height: 8),
                              Text('$changedOutcomes', style: const TextStyle(color: Color(0xFFA78BFA), fontSize: 32, fontWeight: FontWeight.bold)),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 32),
                    // DataTable
                    Card(
                      color: const Color(0xFF1F2937),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      child: SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(minWidth: 800),
                          child: DataTable(
                            headingTextStyle: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
                            dataTextStyle: const TextStyle(color: Colors.white70, fontSize: 14),
                            columns: const [
                              DataColumn(label: Text('Name')),
                              DataColumn(label: Text('Original Score')),
                              DataColumn(label: Text('Fair Score')),
                              DataColumn(label: Text('Changed')),
                              DataColumn(label: Text('Status')),
                            ],
                            rows: docs.map((doc) {
                              final data = doc.data() as Map<String, dynamic>;
                              final name = data['name'] ?? 'Unknown';
                              final originalScore = (data['original_score'] as num?)?.toDouble() ?? 0.0;
                              final fairScore = (data['fair_score'] as num?)?.toDouble() ?? 0.0;
                              final isChanged = data['changed']?.toString().toLowerCase() == 'yes' || data['changed'] == true;
                              final isShortlisted = fairScore > 0.5;

                              return DataRow(cells: [
                                DataCell(Text(name.toString())),
                                DataCell(Text(originalScore.toStringAsFixed(2))),
                                DataCell(Text(fairScore.toStringAsFixed(2))),
                                DataCell(
                                  Text(
                                    isChanged ? 'Yes' : 'No',
                                    style: TextStyle(
                                      color: isChanged ? Colors.greenAccent : Colors.grey,
                                      fontWeight: isChanged ? FontWeight.bold : FontWeight.normal,
                                    ),
                                  ),
                                ),
                                DataCell(
                                  Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                                    decoration: BoxDecoration(
                                      color: isShortlisted ? Colors.green.withOpacity(0.2) : Colors.red.withOpacity(0.2),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Text(
                                      isShortlisted ? 'Shortlisted' : 'Rejected',
                                      style: TextStyle(
                                        color: isShortlisted ? Colors.greenAccent : Colors.redAccent,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ),
                                ),
                              ]);
                            }).toList(),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
