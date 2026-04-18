import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class UploadScreen extends StatefulWidget {
  final Function(String) onUploadComplete;

  const UploadScreen({Key? key, required this.onUploadComplete}) : super(key: key);

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final TextEditingController _modelNameController = TextEditingController(text: "IndicBERT-based hiring model");
  Uint8List? _fileBytes;
  String? _fileName;
  bool _isUploading = false;

  Future<void> _pickFile() async {
    FilePickerResult? result = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv'],
      withData: true,
    );

    if (result != null) {
      setState(() {
        _fileBytes = result.files.single.bytes;
        _fileName = result.files.single.name;
      });
    }
  }

  Future<void> _uploadFile() async {
    if (_fileBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a CSV file first')),
      );
      return;
    }

    setState(() {
      _isUploading = true;
    });

    try {
      final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final storageRef = FirebaseStorage.instance.ref().child('nyaya-uploads/$timestamp.csv');
      
      await storageRef.putData(_fileBytes!);
      
      final docRef = FirebaseFirestore.instance.collection('audit_jobs').doc(timestamp);
      await docRef.set({
        'status': 'pending',
        'model_name': _modelNameController.text,
        'filename': _fileName,
        'created_at': FieldValue.serverTimestamp(),
      });

      widget.onUploadComplete(timestamp);
      
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload failed: $e')),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF080B12),
      body: Center(
        child: Container(
          padding: const EdgeInsets.all(24.0),
          constraints: const BoxConstraints(maxWidth: 600),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Audit Model',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              TextField(
                controller: _modelNameController,
                style: const TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  labelText: 'Model Name',
                  labelStyle: const TextStyle(color: Color(0xFFA78BFA)),
                  enabledBorder: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.grey.shade800),
                  ),
                  focusedBorder: const OutlineInputBorder(
                    borderSide: BorderSide(color: Color(0xFFA78BFA), width: 2),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: _isUploading ? null : _pickFile,
                icon: const Icon(Icons.file_upload),
                label: Text(_fileName == null ? 'Select CSV Dataset' : 'Selected: $_fileName'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1F2937),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
              const SizedBox(height: 32),
              if (_isUploading)
                const Center(
                  child: CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(Color(0xFFA78BFA)),
                  ),
                )
              else
                ElevatedButton(
                  onPressed: _uploadFile,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFA78BFA),
                    foregroundColor: const Color(0xFF080B12),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  child: const Text('Start Audit'),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
