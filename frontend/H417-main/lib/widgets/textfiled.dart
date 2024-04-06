import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class CustomTextField extends StatefulWidget {
  const CustomTextField({
    super.key,
    required this.hintText,
    required this.onValidate,
    this.onSaved,
    this.onChanged,
    this.isPassword = false,
  });

  final String hintText;
  final bool isPassword;
  final void Function(String?)? onChanged;
  final void Function(String?)? onSaved;
  final String? Function(String?) onValidate;

  @override
  State<CustomTextField> createState() => _CustomTextFieldState();
}

class _CustomTextFieldState extends State<CustomTextField> {
  bool obscure = true;
  @override
  Widget build(BuildContext context) {
    return widget.isPassword
        ? Row(
            children: [
              Expanded(
                child: TextFormField(
                  obscureText: obscure,
                  onChanged: widget.onChanged,
                  autocorrect: false,
                  onSaved: widget.onSaved,
                  validator: widget.onValidate,
                  style: GoogleFonts.kadwa(
                    color: Colors.white,
                    fontSize: 14,
                  ),
                  decoration: InputDecoration(
                    hintText: widget.hintText,
                    hintStyle: GoogleFonts.kadwa(
                      color: Colors.white,
                      fontSize: 14,
                    ),
                    filled: true,
                    fillColor: const Color(0xFF010028),
                    border: OutlineInputBorder(
                      borderSide:
                          const BorderSide(width: 2, color: Colors.white),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          const BorderSide(width: 2, color: Colors.white),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderSide:
                          const BorderSide(width: 2, color: Colors.white),
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                ),
              ),
              IconButton(
                onPressed: () {
                  setState(() {
                    obscure = !obscure;
                  });
                },
                icon: Icon(
                  obscure
                      ? Icons.remove_red_eye_outlined
                      : Icons.remove_red_eye,
                ),
              ),
            ],
          )
        : TextFormField(
            onChanged: widget.onChanged,
            autocorrect: false,
            onSaved: widget.onSaved,
            validator: widget.onValidate,
            style: GoogleFonts.kadwa(
              color: Colors.white,
              fontSize: 14,
            ),
            decoration: InputDecoration(
              hintText: widget.hintText,
              hintStyle: GoogleFonts.kadwa(
                color: Colors.white,
                fontSize: 14,
              ),
              filled: true,
              fillColor: const Color(0xFF010028),
              border: OutlineInputBorder(
                borderSide: const BorderSide(width: 2, color: Colors.white),
                borderRadius: BorderRadius.circular(8),
              ),
              enabledBorder: OutlineInputBorder(
                borderSide: const BorderSide(width: 2, color: Colors.white),
                borderRadius: BorderRadius.circular(8),
              ),
              focusedBorder: OutlineInputBorder(
                borderSide: const BorderSide(width: 2, color: Colors.white),
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          );
  }
}
