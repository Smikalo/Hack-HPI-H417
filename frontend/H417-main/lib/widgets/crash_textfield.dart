import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class CrashTextField extends StatelessWidget {
  const CrashTextField({
    super.key,
    required this.hintText,
    this.readOnly = false,
    this.controllerText,
    this.textStyle,
    // required this.helperText,
  });
  final String? controllerText;
  final String hintText;
  final bool readOnly;
  final TextStyle? textStyle;
  // final String helperText;

  @override
  Widget build(BuildContext context) {
    return Material(
      child: TextField(
        style: textStyle,
        controller: TextEditingController(text: controllerText),
        readOnly: readOnly,
        decoration: InputDecoration(
          // helperText: helperText,
          helperStyle: GoogleFonts.kadwa(
            color: const Color(0xFFE2F1F3),
            fontSize: 15,
          ),
          filled: true,
          fillColor: const Color(0xFFE1F0F3),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
            borderSide: BorderSide.none,
          ),
          hintText: hintText,
          hintStyle: GoogleFonts.kadwa(
            color: Colors.black
                .withOpacity(0.5), // Change hint text color if needed
          ),
        ),
      ),
    );
  }
}
