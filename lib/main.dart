import 'package:flutter/material.dart';
import 'screens/upload_screen.dart';
import 'screens/results_screen.dart';
import 'screens/audit_trail_screen.dart';

void main() {
  runApp(const NyayaDashboardApp());
}

class NyayaDashboardApp extends StatelessWidget {
  const NyayaDashboardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Nyaya Bias Audit Dashboard',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.deepPurple,
        colorScheme: const ColorScheme.dark(
          primary: Colors.deepPurple,
          secondary: Colors.tealAccent,
        ),
      ),
      home: const MainNavigationScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    UploadScreen(onUploadComplete: (jobId) {}),
    ResultsScreen(),
    AuditTrailScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Nyaya Dashboard'),
        centerTitle: true,
      ),
      body: _screens[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.upload_file),
            label: 'Upload',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.analytics),
            label: 'Results',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history),
            label: 'Audit Trail',
          ),
        ],
      ),
    );
  }
}
