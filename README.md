# What we built (Frontend Dashboard)
## Day 1 — Flutter Project Setup

   Initialized Flutter web application (nyaya_dashboard) with Android and web platform support
   Configured Firebase connection using FlutterFire CLI linking to nyaya-audit project
   Built 3-screen navigation architecture with Bottom Navigation Bar — Upload, Results, Audit Trail
   Applied dark theme UI with primary color #534AB7 and background #080b12
 
## Day 2 — Core Screens

   Built Upload screen with model name input, CSV file picker and Start Audit button
   Implemented real-time Results screen using Firestore StreamBuilder that auto-updates without refresh
   Results screen displays 4 bias metric cards, CERTIFIED COMPLIANT/BIAS DETECTED verdict banner and Gemini plain English explanation
   Added loading spinner during file upload and error handling with SnackBar notifications

## Day 3 — Visualisation and Audit Trail

   Built interactive t-SNE scatter chart using fl_chart showing applicant embeddings across 4 demographic groups — Brahmin (purple), Dalit (yellow), Hindu (green), Muslim (blue)
   Implemented before/after debiasing toggle to visually demonstrate bias reduction
   Built Audit Trail DataTable reading from Firestore subcollection showing per-applicant name, original score, fair score and shortlisting status
   Added summary line showing total applicants audited and number of outcomes changed
