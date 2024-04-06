import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class BrightButton extends StatelessWidget {
  const BrightButton({super.key, required this.onTap, required this.title});
  final void Function() onTap;
  final String title;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: const Color(0xFFB9C1CB),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
          side: const BorderSide(width: 5, color: Colors.white),
        ),
        elevation: 0,
        padding: EdgeInsets.zero,
        shadowColor: Colors.white,
        minimumSize: const Size(175, 57),
        alignment: Alignment.center,
      ),
      child: Text(
        title,
        style: GoogleFonts.kadwa(
          color: Colors.black,
          fontSize: 24,
          height: 0,
        ),
      ),
    );
  }
}
