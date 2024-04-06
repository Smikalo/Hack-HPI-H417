import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

final themeProvider =StateProvider<ThemeData>((ref) {
  return ThemeData(
  scaffoldBackgroundColor: const Color.fromARGB(255, 1, 0, 40),
  colorScheme: ColorScheme.fromSeed(
    brightness: Brightness.dark,
    seedColor: const Color.fromARGB(255, 16, 12, 136),
  ),
  textTheme: GoogleFonts.latoTextTheme(),
);
},);