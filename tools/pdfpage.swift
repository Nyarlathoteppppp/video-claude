// Render one page of a PDF to a PNG with macOS PDFKit (paper first pages and figures for the videos).
//
//     swiftc -O tools/pdfpage.swift -o pdfpage          # or run it directly: swift tools/pdfpage.swift ...
//     ./pdfpage paper.pdf 0 page1.png 1600              # page index is 0-based; 1600 = longest side in pixels
import Foundation
import PDFKit
import AppKit

let a = CommandLine.arguments
guard a.count == 5, let doc = PDFDocument(url: URL(fileURLWithPath: a[1])), let page = doc.page(at: Int(a[2])!) else {
    print("usage: pdfpage in.pdf pageIndex out.png maxSide"); exit(1)
}
let box = page.bounds(for: .mediaBox)
let scale = CGFloat(Double(a[4])!) / max(box.width, box.height)
let w = Int(box.width * scale), h = Int(box.height * scale)
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h, bitsPerSample: 8, samplesPerPixel: 4,
                           hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
let ctx = NSGraphicsContext(bitmapImageRep: rep)!
NSGraphicsContext.current = ctx
NSColor.white.setFill(); NSRect(x: 0, y: 0, width: w, height: h).fill()   // PDFs have no background of their own
ctx.cgContext.scaleBy(x: scale, y: scale)
page.draw(with: .mediaBox, to: ctx.cgContext)
NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: a[3]))
print("\(a[3]) \(w)x\(h) of \(doc.pageCount) pages")
